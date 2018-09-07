#!/usr/bin/env python

'''
	CryptoKnight
	@gregorydhill
'''

import subprocess

LOGO = "src/logo"

def title():
	print ""
	width = int(subprocess.check_output(['stty', 'size']).split()[1])
	with open(LOGO, "r") as logo:
		title = logo.readlines()
		for line in title:
			print (((width/2)-(34/2)) * " ") + line,
	print "@gregorydhill \n\n".center(width)
