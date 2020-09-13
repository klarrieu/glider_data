import os

"""
This script converts Slocum glider binary data (.dbd, .ebd, and .cac files) into ASCII format for further processing.
"""

# path with binary glider data (.dbd, .ebd)
data_path = os.path.join(os.path.dirname(__file__), 'data\\glider\\binary')
# path with dbd2asc and dbamerge executables
exe_path = os.path.join(os.path.dirname(__file__), 'bin')


# append executables to environment path
os.environ['PATH'] += os.pathsep + exe_path

for dir, subdirs, files in os.walk(data_path):
    for f in files:
        # find .dbd files
        if f.lower().endswith('.dbd'):
            # account for upper or lowercase name extension conventions
            if f.endswith('.dbd'):
                ext = '.dbd'
                comp_ext = '.ebd'
            else:
                ext = '.DBD'
                comp_ext = '.EBD'

            # check if complementary file exists
            comp_file = f.replace(ext, comp_ext)
            if os.path.isfile(os.path.join(dir, comp_file)):
                print('processing %s, %s' % (f, comp_file))
                # convert both files from binary to ascii
                os.system('dbd2asc %s >data.dba' % os.path.join(dir, f))
                print('converted .dbd')
                os.system('dbd2asc %s >data.eba' % os.path.join(dir, comp_file))
                print('converted .ebd')
                # merge complementary files into single file
                os.system('dba_merge data.dba data.eba | dba2_glider_data')
                print('merged ascii data')
