import pstats
from pstats import SortKey

p = pstats.Stats('out.profile')
p.strip_dirs().sort_stats(-1).print_stats()
