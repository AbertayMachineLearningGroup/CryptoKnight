#!/usr/bin/env python

'''
	CryptoKnight

	@gregorydhill
'''
import urllib, tarfile, os, sys, shutil, filecmp, pwd, grp

URL = 'https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.4-97438-gf90d1f746-gcc-linux.tar.gz'
FILE = URL.split('/')[-1]
BASE = os.path.dirname(os.path.realpath(__file__))
PIN = os.path.join(BASE, "../pin_tool")
SRC = os.path.join(BASE, "Trace")
TOOL = "source/tools/CryptoKnight/"

def alive():
	if (not exists(os.path.join(PIN))):
		return False
	elif (not exists(os.path.join(PIN, TOOL))):
		return False
	else: return True

def exists(path):
	if (not os.path.isdir(path)):
		return False
	else:
		return True

def install_pin():
	if (not exists(os.path.join(PIN))):
		print("[+] Downloading Pin Framework...")
		urllib.urlretrieve(URL, FILE)
		print ("[+] Downloaded: %s" % (FILE))
		tar = tarfile.open(FILE)
		tar.extractall()
		sub_name = os.path.commonprefix(tar.getnames())
		tar.close()
		os.remove(FILE)
		os.rename(sub_name, os.path.join(PIN))
	if (not exists(os.path.join(PIN, TOOL))):
		print("[+] Copying & Compiling PinTools")
	else:
		shutil.rmtree(os.path.join(PIN, TOOL))
		print("[+] Updating PinTools")
	os.mkdir(os.path.join(PIN, TOOL))
	for files in os.listdir(os.path.join(BASE, SRC)):
		shutil.copyfile(os.path.join(BASE, SRC, files), os.path.join(PIN, TOOL, files))
	os.system("cd " + os.path.join(PIN, TOOL) + "; make all > /dev/null")

def setup():
	try:
		try:
			import pip
		except ImportError:
			os.system("apt-get install python-pip")
		try:
			import torch
		except ImportError:
			print("[!] Install pytorch manually.")
			return False

		install_pin()

		#os.system("apt-get install libssl-dev > /dev/null")
		#os.system("mount --bind src/Trace test/src")
		#os.system("mount --bind pin_tool test/pin_tool")
		#os.system("pip install -r " + os.path.join(BASE, "../requirements.txt") + " > /dev/null")

	except OSError:
		print "[!] Requires administrator privileges."
		return False
	return True
	
if __name__ == "__main__":
	setup()
