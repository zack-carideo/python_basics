#find where python is installed 
where python

#upgrade package 
pip install numpy -U

#upgrade pip
python -m pip install --upgrade pip

#temporarly add python to system path (great if u dont have admin rights) 
set Path = C:\Program Files\Python37\python.exe;%Path%


#identify dependencies of any module (and the dependency versions) 
from pip._vendor import pkg_resources
_package_name = 'keras_bert'
_package = pkg_resources.working_set.by_key[_package_name]
print([str(r) for r in _package.requires()])

#list files in directory 
def list_files(dir_path):
    files = []
    for name in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, name)):
        files.append(name)
    return files