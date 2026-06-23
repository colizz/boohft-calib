import multiprocessing
from multiprocessing.pool import ThreadPool
import pickle
import os
import signal
import copy
from tqdm.auto import tqdm
from coffea import processor
from coffea.nanoevents import BaseSchema
import uproot
from uproot.source.file import MemmapSource


class _EventBatch(object):
    """Minimal NanoEvents-like wrapper for coffea versions without Runner."""

    def __init__(self, arrays, metadata):
        self._arrays = arrays
        self.metadata = metadata

    def __len__(self):
        return len(self._arrays)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._arrays[item]
        return _EventBatch(self._arrays[item], self.metadata)

    def __getattr__(self, name):
        if name in self._arrays.fields:
            return self._arrays[name]
        raise AttributeError(name)

    @property
    def fields(self):
        return self._arrays.fields


def _merge_outputs(left, right):
    if left is None:
        return copy.deepcopy(right)
    if isinstance(left, dict):
        for key, value in right.items():
            left[key] = _merge_outputs(left.get(key), value)
        return left
    try:
        left_view = left.view(flow=True)
        right_view = right.view(flow=True)
        if hasattr(left_view, 'value') and hasattr(left_view, 'variance'):
            left_view.value += right_view.value
            left_view.variance += right_view.variance
        else:
            left_view[...] += right_view
        return left
    except AttributeError:
        left += right
    return left


def _run_dataset_fallback(arg):
    dataset, files, branches, processor_cls, processor_kwargs, chunksize = arg
    processor_instance = processor_cls(**processor_kwargs)
    out = None
    for fname in files:
        with uproot.open(fname + ':Events', handler=MemmapSource) as tree:
            available = set(tree.keys())
            dataset_branches = branches
            if dataset_branches is not None:
                dataset_branches = [branch for branch in dataset_branches if branch in available]
            for start in range(0, tree.num_entries, chunksize):
                stop = min(start + chunksize, tree.num_entries)
                arrays = tree.arrays(dataset_branches, entry_start=start, entry_stop=stop, library='ak')
                events = _EventBatch(arrays, {'dataset': dataset})
                out = _merge_outputs(out, processor_instance.process(events))
    return out

class ProcessingUnit(object):
    r"""A processing unit including
        (1) first runs the standard coffea job;
        (2) stores results into the root file in a custom way;
        (3) make custom plots and write to a webpage for easy visualization.
    """

    def __init__(self, job_name, fileset=None, processor_cls=None, **kwargs):
        self.job_name = job_name
        self.fileset = fileset
        self.processor_cls = processor_cls
        self.processor_kwargs = kwargs.pop('processor_kwargs', dict())
        self.workers = kwargs.pop('workers', 8)


    def preprocess(self):
        pass


    def initalize_processor(self):
        if self.processor_cls is None:
            # Do not run the coffea job
            return
        
        self.processor_instance = self.processor_cls(**self.processor_kwargs)


    def run_coffea_job(self):
        if self.processor_cls is None:
            # Do not run the coffea job
            return

        self.initalize_processor()
        branches = sorted(getattr(self.processor_instance, 'required_branches', [])) or None
        if not hasattr(processor, 'Runner') or not hasattr(processor, 'FuturesExecutor'):
            self.run_coffea_job_fallback(branches)
            return

        fileset = {}
        for dataset, files in self.fileset.items():
            dataset_branches = branches
            if branches is not None:
                with uproot.open(files[0] + ':Events', handler=MemmapSource) as tree:
                    available = set(tree.keys())
                dataset_branches = [branch for branch in branches if branch in available]
            fileset[dataset] = {'files': files, 'preload': dataset_branches}
        executor = processor.FuturesExecutor(
            workers=self.workers,
            desc='coffea datasets',
        )
        runner = processor.Runner(
            executor=executor,
            schema=BaseSchema,
            chunksize=200000,
        )
        self.result = runner(
            fileset,
            treename='Events',
            processor_instance=self.processor_instance,
            uproot_options={'handler': MemmapSource},
        )


    def run_coffea_job_fallback(self, branches):
        tasks = [
            (dataset, files, branches, self.processor_cls, self.processor_kwargs, 200000)
            for dataset, files in self.fileset.items()
        ]
        if self.workers == 1:
            result = None
            for task in tqdm(tasks, total=len(tasks), desc='uproot datasets'):
                result = _merge_outputs(result, _run_dataset_fallback(task))
            self.result = result
            return

        with ThreadPool(processes=self.workers) as pool:
            result = None
            for partial in tqdm(pool.imap_unordered(_run_dataset_fallback, tasks), total=len(tasks), desc='uproot datasets'):
                result = _merge_outputs(result, partial)
        self.result = result


    def load_pickle(self, attrname: str):
        if not os.path.isfile(os.path.join(self.outputdir, attrname + '.pickle')):
            raise FileNotFoundError('Cannot find ' + os.path.join(self.outputdir, attrname + '.pickle'))

        with open(os.path.join(self.outputdir, attrname + '.pickle'), 'rb') as f:
            setattr(self, attrname, pickle.load(f))


    def postprocess(self):
        pass


    def make_webpage(self):
        pass


    def launch(self, skip_coffea=False):
        r"""Launch the processing unit"""
        self.preprocess()
        if not skip_coffea:
            self.run_coffea_job()
        else: # skip coffea step
            if self.processor_cls is not None:
                self.initalize_processor()
                self.load_pickle('result')

        self.postprocess()
        self.make_webpage()


class StandaloneMultiThreadedUnit(object):
    r"""Holds a standalone multi-threaded unit to book and submit multiple processes.
        Can use local resource or batch resources depending on the config. 
    """

    def __init__(self, **kwargs):

        self.workers = kwargs.pop('workers', 8)
        self.use_unordered_mapping = kwargs.pop('use_unordered_mapping', False)

        # book multi tasks handler by multiprocessing.Pool
        def init_worker():
            # allow the subprocess to be notified by the interrupt signal
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.pool = multiprocessing.Pool(processes=self.workers, initializer=init_worker)
        self.args = []
    
    def book(self, arg: tuple):
        # accept one argument only
        self.args.append(arg)

    def run(self, func):
        try:
            imap = self.pool.imap if not self.use_unordered_mapping else self.pool.imap_unordered
            result = list(tqdm(imap(func, self.args), total=len(self.args)))
        except KeyboardInterrupt:
            self.pool.terminate()
            self.pool.join()
            return None
        except Exception:
            self.pool.terminate()
            self.pool.join()
            raise
        else:
            self.pool.close()
            self.pool.join()
            return result
