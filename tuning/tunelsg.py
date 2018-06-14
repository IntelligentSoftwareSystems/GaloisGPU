#!/usr/bin/env python

# save [name] -> collect all tuning headers and save them in tuning/[name] if it does not exist, -f to force
# load [name] -> collect all tuning headers in tuning/[name] and put them in appropriate places, backing up existing files when doing so

import sys
import os
import shutil
from optparse import OptionParser
MAX_BACKUPS = 5

# find -iname '*_tuning.h' | awk -F/ '{printf "'"'"'%s'"'"': '"'"'%s'"'"',\n", $2, $3}'
APPS = {'bfs': 'bfs_wlc_tuning.h',
        'bh': 'bh_tuning.h',
        'dmr': 'dmr_tuning.h',
        'mst': 'mst_tuning.h',
        'pta': 'pta_tuning.h',
        'sp': 'newsp_tuning.h',
        'sssp': 'sssp_wlc_tuning.h'}

APP_DIR = "../apps/"
TUNING_DIR = "./configs"

def fixup_paths():
    global APP_DIR, TUNING_DIR

    my_path = os.path.dirname(sys.argv[0])
    APP_DIR = os.path.join(my_path, APP_DIR)
    TUNING_DIR = os.path.join(my_path, TUNING_DIR)

def get_tuning_include_files():
    out = []
    for a in APPS:
        out.append(os.path.join(APP_DIR, a, APPS[a]))

    return out

def backup_file(fname):
    # if fname exists
    #   delete fname.~MAX_BACKUPS~
    #   copy fname.~X~ to fname.~X+1~
    #   copy fname to fname.~1~

    def _backup_name(suffix):
        return "%s.~%d~" % (fname, suffix)

    if os.path.exists(fname):
        if os.path.exists(_backup_name(MAX_BACKUPS)):
            os.unlink(_backup_name(MAX_BACKUPS))

        for i in range(MAX_BACKUPS-1, 0, -1):
            if os.path.exists(_backup_name(i)):
                os.rename(_backup_name(i), _backup_name(i+1))

        os.rename(fname, _backup_name(1))

def copy_file(src, dst):
    print "'%s' -> '%s'" % (src, dst)
    shutil.copyfile(src, dst)

def save(dst, force = False):
    dstdir = os.path.join(TUNING_DIR, dst)

    if os.path.exists(dstdir):
        if not force:
            print >>sys.stderr, "ERROR: %s exists, will not overwrite. Use -f to force." % (dstdir,)
            return False
        else:
            print >>sys.stderr, "WARNING: %s exists, files will be overwritten." % (dstdir,)
    else:
        os.mkdir(dstdir)
        
    fl = get_tuning_include_files()

    for src in fl:
        dst = os.path.join(dstdir, os.path.basename(src))
        copy_file(src, dst)

    return True

def load(src):
    srcdir = os.path.join(TUNING_DIR, src)

    if not os.path.exists(srcdir):
        print >>sys.stderr, "ERROR: %s does not exist." % (srcdir, )
        return False

    fl = get_tuning_include_files()

    for dst in fl:
        src = os.path.join(srcdir, os.path.basename(dst))
        backup_file(dst)
        copy_file(src, dst)
        
    return True

parser = OptionParser("Usage: %prog [options] configuration")
parser.add_option("", "--load", dest="cmd", action="store_const", const="load",
                  help="load configuration")
parser.add_option("", "--save", dest="cmd", action="store_const", const="save",
                  help="save configuration")

parser.add_option("-f", "--force", dest="force", action="store_true", default=False,
                  help="Force overwrites during save")

(options, args) = parser.parse_args()
        
if len(args) != 1:
    parser.error("need configuration to load/save")
else:
    CONFIG = args[0]

fixup_paths()

if options.cmd == "load":
    load(CONFIG)
elif options.cmd == "save":
    save(CONFIG, force=options.force)
else:
    parser.error("--load/--save must be specified")
