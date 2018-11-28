import pip
from subprocess import call

call("python3 -m spacy download en", shell=True)

# Updates all installed pip3 python packages
# for dist in pip.get_installed_distributions():
#     print("pip3 install " + dist.project_name + " -U")
#     call("pip3 install " + dist.project_name + " -U", shell=True)
