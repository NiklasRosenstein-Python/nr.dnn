
import setuptools

def readme():
  with open('README.md', encoding='utf8') as fp:
    return fp.read()

def requirements():
  with open('requirements.txt') as fp:
    return fp.readlines()

setuptools.setup(
  name = 'nr.dnn',
  version = '1.0.0',
  author = 'Niklas Rosenstein',
  author_email = 'rosensteinniklas@gmail.com',
  description = 'git',
  long_description = readme(),
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/NiklasRosenstein-Python/nr.dnn',
  license = 'MIT',
  install_requires = requirements(),
  packages = setuptools.find_packages('src'),
  package_dir = {'': 'src'}
)
