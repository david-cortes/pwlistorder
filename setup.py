from distutils.core import setup
setup(
  name = 'pwlistorder',
  packages = ['pwlistorder'],
  install_requires=[
   'pandas',
   'numpy',
   'cvxpy'
],
  version = '0.1',
  description = 'Ordering lists based on aggregated pairwise preferences',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/pwlistorder',
  download_url = 'https://github.com/david-cortes/pwlistorder/archive/0.1.tar.gz',
  keywords = ['pairwise ranking', 'list ordering', 'kwik-sort'],
  classifiers = [],
)