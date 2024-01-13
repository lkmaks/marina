import threading

class WorkerThread(threading.Thread):
    def __init__(self, wcfg, ncfg, stats, print_lock=None, model=None):
        threading.Thread.__init__(self)
        self.wcfg = wcfg
        self.ncfg = ncfg
        self.stats = stats
        self.print_lock = print_lock

        if model is None:
            self.model = utils.getModel(ncfg.model_name, wcfg.train_set_full, wcfg.device)
            utils.setupAllParamsRandomly(self.model)
        else:
            self.model = copy_module(model)

        self.model = self.model.to(wcfg.device)  # move model to device
        wcfg.model = self.model