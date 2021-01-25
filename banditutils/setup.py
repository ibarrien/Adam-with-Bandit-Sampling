from distutils.core import setup

#https://caremad.io/posts/2013/07/setup-vs-requirement/

setup(name='banditutils',		#TODO: change this to be a good package name
	version='1.01',
	description='Bandit Sampling Core',
	author='ibarrien, Manny Ko',
	author_email='man960@hotmail.com',
	#url='https://www.python.org/sigs/distutils-sig/',
	packages=[
		'banditutils',
		'banditutils.layers',
		'banditutils.utils',
	],
	install_requires=[
		"tensorflow==1.13.1",
		"keras==2.2.4",
		"blinker",
		"matplotlib",
	],
)

