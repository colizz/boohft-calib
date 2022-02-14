from coffea.nanoevents import BaseSchema
from coffea import processor

import multiprocessing
import pickle
import os
import signal
from tqdm.auto import tqdm

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
        
        if not hasattr(self, 'processor_instance'):
            self.initalize_processor()

        self.result = processor.run_uproot_job(
            fileset=self.fileset,
            treename="Events",
            processor_instance=self.processor_instance,
            executor=processor.futures_executor,
            executor_args={"schema": BaseSchema, "workers": self.workers},
        )


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
            return result
        except KeyboardInterrupt:
            self.pool.terminate()
            self.pool.join()
            return None
