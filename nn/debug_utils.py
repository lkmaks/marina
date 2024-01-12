
def dbgprint(print_lock, wcfg, *args):
    print_lock.acquire()
    print(f"Worker {wcfg.worker_id}/{wcfg.total_workers}:", *args, flush = True)
    print_lock.release()

def rootprint(print_lock, *args):
    print_lock.acquire()
    print(f"Master: ", *args, flush = True)
    print_lock.release()