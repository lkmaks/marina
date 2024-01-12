from utils import get_all_log_files
import subprocess as sb

fs_ds = get_all_log_files()

lst = ['python', 'show.py'] + [e[1] for e in fs_ds]

sb.run(lst)
