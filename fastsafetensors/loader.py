# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

from . import cpp as fstcpp
from .common import (
    SafeTensorsMetadata,
    TensorFrame,
    get_device_numa_node,
    init_logger,
    set_debug,
)
from .copier import CopierConstructFunc, CopierType, create_copier_constructor
from .file_buffer import FilesBufferOnDevice
from .frameworks import TensorBase, get_framework_op
from .st_types import Device, DType
from .tensor_factory import LazyTensorFactory

gl_set_numa = False

loaded_library = False

logger = init_logger(__name__)


class BaseSafeTensorsFileLoader:
    r"""Base class for loading .safetensors files lazily.

    Args:
        pg (Optional[Any]): Process group-like objects for distributed loading.
                           Use None for single device use-cases.
        device (Device): Target device where tensors will be loaded (CPU, CUDA, etc.).
        copier_constructor: Constructor function for creating file copier objects.
        set_numa (bool): Whether to set NUMA node affinity for optimized memory access.
        disable_cache (bool): Whether to disable caching of loaded tensors.
        debug_log (bool): Enable detailed debug logging.
        framework (str): Deep learning framework to use ("pytorch" or "paddle").
    """

    def __init__(
        self,
        pg: Optional[Any],
        device: Device,
        copier_type: CopierType,
        set_numa: bool = True,
        disable_cache: bool = True,
        framework="pytorch",
        **kwargs,
    ):
        self.framework = get_framework_op(framework)
        self.pg = self.framework.get_process_group(pg)
        self.device = device
        self.meta: Dict[str, Tuple[SafeTensorsMetadata, int]] = {}
        self.frames = OrderedDict[str, TensorFrame]()
        self.disable_cache = disable_cache
        self.init_numa(set_numa)
        self.copier_constructor: CopierConstructFunc = create_copier_constructor(
            copier_type=copier_type,
            device=device,
            **kwargs,
        )

    def init_numa(self, set_numa: bool = True):
        global gl_set_numa
        if not gl_set_numa and set_numa:
            node = get_device_numa_node(self.device.index)
            if node is not None:
                fstcpp.set_numa_node(node)
            gl_set_numa = True

    def reset(self):
        self.frames = {}
        self.meta = {}

    def close(self):
        self.reset()
        del self.copier_constructor

    def get_keys(self) -> List[str]:
        return list(self.frames.keys())

    def get_shape(self, tensor_name: str) -> List[int]:
        return self.frames[tensor_name].shape

    def add_filenames(self, filenames: Dict[int, List[str]]):
        """
        Register files to ranks to be copied at copy_file_to_device().
        """
        # shuffle files in a round-robin fashion to avoid OoM
        rank_next_idx = {rank: 0 for rank in filenames.keys()}
        completed = 0
        while completed < len(filenames.keys()):
            completed = 0
            for rank in filenames.keys():
                next_idx = rank_next_idx[rank]
                if next_idx < len(filenames[rank]):
                    realpath = filenames[rank][next_idx]  # os.path.realpath(filename)
                    metadata = SafeTensorsMetadata.from_file(realpath, self.framework)
                    self.meta[realpath] = (metadata, rank)
                    self.frames.update(metadata.tensors)
                    if rank == self.pg.rank():
                        logger.debug(
                            "add_filenames %d: path=%s", len(self.meta), realpath
                        )
                    rank_next_idx[rank] = next_idx + 1
                else:
                    completed += 1

    def copy_files_to_device(
        self,
        dtype: DType = DType.AUTO,
        use_buf_register: bool = True,
        max_copy_block_size: int = 16 * 1024 * 1024 * 1024,
    ) -> FilesBufferOnDevice:
        """
        trigger copying all the files to device buffers.
        At this moment, we do not instantiate tensors but just creating copies at device buffers with or without GDS.
        Users can instantiate and/or partition tensors with FilesBufferOnDevice returned by this function.
        """
        self.framework.set_device(self.device)

        need_wait: List[LazyTensorFactory] = []
        factories: Dict[int, List[LazyTensorFactory]] = {}
        for i in range(0, self.pg.size()):
            factories[i] = []

        factory_idx_bits = math.ceil(math.log2(len(self.meta) + 1))
        lidx = 1
        for _, (meta, rank) in sorted(self.meta.items(), key=lambda x: x[0]):
            copier = self.copier_constructor(meta, self.device, self.framework)
            self_rank = self.pg.rank() == rank
            factory = LazyTensorFactory(
                meta,
                self.device,
                rank,
                self_rank,
                factory_idx_bits,
                lidx,
                copier,
                self.framework,
                disable_cache=self.disable_cache,
            )
            factory.submit_io(use_buf_register, max_copy_block_size)
            factories[rank].append(factory)
            if self_rank:
                need_wait.append(factory)
            lidx += 1
        for factory in need_wait:
            factory.wait_io(dtype=dtype, noalign=False)
        return FilesBufferOnDevice(factories, pg=self.pg, framework=self.framework)


class SafeTensorsFileLoader(BaseSafeTensorsFileLoader):
    r"""Load .safetensors files lazily.

    Args:
        devcie (str): target device.
        pg (Optional[Any]): process group-like objects for distributed. None for single GPU use-cases.
        bbuf_size_kb (int): bounce buffer size for file copies.
        max_threads (int): maximum number of threads for memory copies.
        nogds (bool): if True, trun off GDS and fallback to pread with bounce buffer.
        debug_log (bool): enable debug logs.

    Examples:
        >> from fastsafetensors import SafeTensorsFileLoader
        >> src_files = download(target_dir, "gpt2")
        >> loader = SafeTensorsFileLoader(Device("cpu"), nogds=True, debug_log=True)
        >> loader.add_filenames({0: src_files})
        >> bufs = loader.copy_files_to_device()
        >> print(bufs.get_tensor(loader.get_keys()[0]))
        >> loader.close()
    """

    def __init__(
        self,
        pg: Optional[Any],
        device: str = "cpu",
        bbuf_size_kb: int = 16 * 1024,
        max_threads: int = 16,
        nogds: bool = False,
        set_numa: bool = True,
        disable_cache: bool = True,
        debug_log: bool = False,
        framework="pytorch",
        **kwargs,
    ):
        self.framework = get_framework_op(framework)
        self.pg = self.framework.get_process_group(pg)
        self.device = self.framework.get_device(device, self.pg)

        fstcpp.set_debug_log(debug_log)
        if nogds:
            copier_type = "nogds"
        else:
            copier_type = "gds"
        super().__init__(
            pg,
            self.device,
            copier_type,
            set_numa,
            disable_cache,
            framework,
            bbuf_size_kb=bbuf_size_kb,
            max_threads=max_threads,
            **kwargs,
        )


class fastsafe_open:
    """
    Opens a safetensors lazily and returns tensors as asked
    This is an enhanced version of safe_open in the original safetensors library to consume file list

    Args:
        filenames (:obj:`str`|`list[str]`|`dict[int, str]`): The filename(s) or rank-file map to open
        framework (:obj:`str`): `pt`, `pytorch`, and `paddle` are only supported currently
        device (:obj:`str`, defaults to :obj:`"cpu"`): The device on which you want the tensors.
    """

    def __init__(
        self,
        filenames: Union[str, List[str], Dict[int, List[str]]],
        framework: str = "pt",
        pg: Optional[Any] = None,
        device: str = "cpu",
        nogds: bool = False,
        debug_log: bool = False,
        max_copy_block_size: int = 16 * 1024 * 1024 * 1024,
    ):
        self.loader = SafeTensorsFileLoader(
            pg, device, nogds=nogds, debug_log=debug_log, framework=framework
        )
        file_dict: Dict[int, List[str]] = {}
        if isinstance(filenames, str):
            file_dict = {0: [filenames]}
        if isinstance(filenames, list):
            file_dict = {0: filenames}
        elif isinstance(filenames, dict):
            file_dict = filenames
        self.loader.add_filenames(file_dict)
        self.fb = self.loader.copy_files_to_device(
            max_copy_block_size=max_copy_block_size
        )

    def metadata(self) -> Dict[str, Dict[str, str]]:
        ret = {}
        for filename, (metadata, _) in self.loader.meta.items():
            ret[filename] = metadata.metadata
        return ret

    def keys(self) -> List[str]:
        return list(self.fb.key_to_rank_lidx.keys())

    def get_tensor_wrapped(self, name: str) -> TensorBase:
        return self.fb.get_tensor_wrapped(name)

    def get_tensor(self, name: str) -> Any:
        return self.get_tensor_wrapped(name).get_raw()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.fb:
            self.fb.close()
        if self.loader:
            self.loader.close()


class fastsafe_open_streaming:
    """Load safetensors with chunked streaming I/O to avoid 2x VRAM peak.

    Unlike fastsafe_open which loads the entire file into a single GPU buffer
    (causing peak VRAM = 2 * model_size when cloning), this class loads the file
    in fixed-size chunks. For each chunk, it reads only that portion into a
    temporary GPU buffer, clones the tensors that fall within the chunk, and
    immediately frees the chunk buffer.

    Peak VRAM overhead = model_size + chunk_size (instead of 2 * model_size).

    Args:
        filename (str): Path to the .safetensors file.
        framework (str): ``pt``, ``pytorch``, or ``paddle``.
        device (str): Target device (e.g., ``cuda:0``).
        nogds (bool): If True, disable GDS and use pread fallback.
        chunk_size (int): Size of each I/O chunk in bytes (default 1 GB).
        debug_log (bool): Enable debug logging.

    Examples:
        >>> with fastsafe_open_streaming("model.safetensors", device="cuda:0", nogds=True) as f:
        ...     sd = {k: f.get_tensor(k) for k in f.keys()}
    """

    def __init__(
        self,
        filename: str,
        framework: str = "pt",
        device: str = "cpu",
        nogds: bool = False,
        chunk_size: int = 1 * 1024 * 1024 * 1024,  # 1 GB
        debug_log: bool = False,
    ):
        fstcpp.set_debug_log(debug_log)
        self._framework = get_framework_op(framework)
        self._device = self._framework.get_device(device, self._framework.get_process_group(None))
        self._filename = filename
        self._chunk_size = chunk_size
        self._nogds = nogds

        # Parse header to get tensor metadata without loading data
        self._metadata = SafeTensorsMetadata.from_file(filename, self._framework)
        self._header_length = self._metadata.header_length
        self._file_data_size = self._metadata.size_bytes - self._header_length

        # Sort tensors by offset for sequential I/O access
        self._sorted_tensors = sorted(
            self._metadata.tensors.items(),
            key=lambda x: x[1].data_offsets[0]
        )

        # Pre-load all tensors via chunked streaming
        self._tensors: Dict[str, Any] = {}
        self._load_chunked()

    def _load_chunked(self):
        """Load all tensors using chunked I/O to minimize peak VRAM."""
        fd = os.open(self._filename, os.O_RDONLY, 0o644)
        try:
            chunk_start = 0
            while chunk_start < self._file_data_size:
                chunk_end = min(chunk_start + self._chunk_size, self._file_data_size)
                chunk_len = chunk_end - chunk_start

                # Allocate a temporary GPU buffer for this chunk only
                chunk_buf = self._framework.alloc_tensor_memory(chunk_len, self._device)

                # Read this chunk from disk into the buffer
                file_offset = self._header_length + chunk_start
                bytes_read = os.pread(fd, chunk_len, file_offset)
                if len(bytes_read) != chunk_len:
                    raise IOError(
                        f"Short read: expected {chunk_len} bytes at offset {file_offset}, "
                        f"got {len(bytes_read)}"
                    )

                # Copy from host to device buffer using the C++ memcpy
                # We need to use the nogds reader for host-to-device transfer
                self._copy_host_to_device(chunk_buf, bytes_read, chunk_len)

                # Find tensors that fall entirely within this chunk
                for tensor_name, frame in self._sorted_tensors:
                    if tensor_name in self._tensors:
                        continue  # already loaded in a previous chunk

                    t_start = frame.data_offsets[0]
                    t_end = frame.data_offsets[1]

                    # Check if tensor falls within current chunk
                    if t_start >= chunk_start and t_end <= chunk_end:
                        # Create a DLPack view into the chunk buffer at the right offset
                        from .dlpack import from_cuda_buffer
                        local_offset = t_start - chunk_start
                        dev_ptr = chunk_buf.get_base_address() + local_offset

                        disk_dtype = self._framework.as_workaround_dtype(frame.dtype)
                        dl_tensor = from_cuda_buffer(
                            dev_ptr, frame.shape, frame.strides, disk_dtype, self._device
                        )
                        t = self._framework.from_dlpack(dl_tensor, self._device, disk_dtype)
                        if disk_dtype != frame.dtype:
                            t = t.view(frame.dtype)

                        # Clone into independent memory and store
                        cloned = t.clone().detach()
                        self._tensors[tensor_name] = cloned.get_raw()

                # Free the chunk buffer immediately
                self._framework.free_tensor_memory(chunk_buf, self._device)

                chunk_start = chunk_end

            # Handle any tensors that spanned chunk boundaries
            for tensor_name, frame in self._sorted_tensors:
                if tensor_name not in self._tensors:
                    t_start = frame.data_offsets[0]
                    t_end = frame.data_offsets[1]
                    t_size = t_end - t_start

                    # Allocate a buffer exactly for this tensor
                    tensor_buf = self._framework.alloc_tensor_memory(t_size, self._device)
                    file_offset = self._header_length + t_start
                    bytes_read = os.pread(fd, t_size, file_offset)
                    if len(bytes_read) != t_size:
                        raise IOError(
                            f"Short read for spanning tensor {tensor_name}: "
                            f"expected {t_size}, got {len(bytes_read)}"
                        )
                    self._copy_host_to_device(tensor_buf, bytes_read, t_size)

                    from .dlpack import from_cuda_buffer
                    dev_ptr = tensor_buf.get_base_address()
                    disk_dtype = self._framework.as_workaround_dtype(frame.dtype)
                    dl_tensor = from_cuda_buffer(
                        dev_ptr, frame.shape, frame.strides, disk_dtype, self._device
                    )
                    t = self._framework.from_dlpack(dl_tensor, self._device, disk_dtype)
                    if disk_dtype != frame.dtype:
                        t = t.view(frame.dtype)
                    cloned = t.clone().detach()
                    self._tensors[tensor_name] = cloned.get_raw()
                    self._framework.free_tensor_memory(tensor_buf, self._device)

        finally:
            os.close(fd)

    def _copy_host_to_device(self, gbuf, host_bytes: bytes, length: int):
        """Copy host bytes into a gds_device_buffer via cudaMemcpy."""
        import ctypes
        # Allocate pinned host buffer, copy data, then device memcpy
        host_ptr = fstcpp.cpu_malloc(length)
        try:
            ctypes.memmove(host_ptr, host_bytes, length)
            # Use the C++ level memmove: dst_buf.memmove(dst_off, src_off, tmp_buf, length)
            # We create a temporary host gds_device_buffer wrapping the host_ptr
            host_buf = fstcpp.gds_device_buffer(host_ptr, length, False)
            gbuf.memmove(0, 0, host_buf, length)
        finally:
            fstcpp.cpu_free(host_ptr)

    def metadata(self) -> Dict[str, str]:
        return self._metadata.metadata if self._metadata.metadata else {}

    def keys(self) -> List[str]:
        return list(self._tensors.keys())

    def get_tensor(self, name: str) -> Any:
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found. Available: {list(self._tensors.keys())[:5]}...")
        return self._tensors[name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._tensors = {}
