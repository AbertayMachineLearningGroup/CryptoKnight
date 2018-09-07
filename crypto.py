#!/usr/bin/env python

'''
	CryptoKnight
	@gregorydhill
'''

import os, sys, csv, shutil, datetime, argparse, subprocess, signal, math
import numpy as np
from src.generate import Generate
from src.setup import alive, setup
from src.title import title

BASE = os.path.dirname(os.path.realpath(__file__))
HEAD = os.path.join(BASE, "data")
FLAGS = os.path.join(HEAD, "config/flags")
LABELS = os.path.join(HEAD, "config/labels")
POOL = os.path.join(HEAD, "config/pool")
FABRICATOR = os.path.join(HEAD, "fab")
DIST = os.path.join(HEAD, "dist")
PIN = "./pin_tool"
PROGRESS = int(subprocess.check_output(['stty', 'size']).split()[1])-10

def signal_handler(signal, frame):
	print("\n\n[!] Killing remaining processes and exiting.")
	os.system('stty echo; setterm -cursor on;')
	sys.exit(0)

def update_progress(count, total, distribution):
	# update drawing progress
	filled_len = int(round(PROGRESS * count / float(total)))
	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (PROGRESS - filled_len)
	sys.stdout.write("[%s] %s%s\r" % (bar, percents, "%"))
	sys.stdout.flush()

def run_analysis(samples, distribution):
	print("[+] Drawing " + distribution + "...")

	# create list of all executables
	variants = []
	for s in samples:
		os.system("cd " + s[1] + "/; make all > /dev/null 2>&1")
		for exe in os.listdir(os.path.join(s[1])):
			if exe.endswith('.exe'):
				variants.append([s[0], os.path.join(s[1], exe)])

	for i, v in enumerate(variants):
		update_progress(i, len(variants), distribution)

		# run pintool in killable subprocess
		cmd = [PIN + "/pin", "-t", PIN + "/source/tools/CryptoKnight/obj-intel64/CryptoTrace.so"]
		cmd.extend(["-v", "1", "-o", distribution, "-l", str(v[0]), "--", v[1]])
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		try:
			p.wait()
		except KeyboardInterrupt:
			try:
				p.terminate()
			except OSError:
				pass
		p.wait()

	update_progress(len(variants), len(variants), distribution)	# completion
	print("\n")

def extend(sets, multiply):
	for dist in sets:
		with open(os.path.join(DIST, dist[0])) as l:
			lines = l.read().splitlines()
			dset = open(data_parent + data_set, 'w+')
			for i in range(0, multiply):
				for vector in lines:
					dset.write(str(vector)+"\n")
			dset.close()

def train_model(sets, tune):
	try:
		train = os.path.join(DIST, sets[0][0])
		test = os.path.join(DIST, sets[1][0])
		if not tune:
			print("\n[+] Training...")
			os.system("cd src/Model/; python dcnn.py --train " + train + " --test " + test)
		else:
			print("\n[+] Tuning...")
			os.system("cd src/Model/; python dcnn.py --train " + train + " --test " + test + " --tune " + str(tune))
	except IndexError:
		print("[!] Training requires a distribution.")

sys.stdout.write("\n")

def main(argv):
	parser = argparse.ArgumentParser(description='CryptoKnight - Generation & Extraction')
	parser.add_argument('-d', '--dist', type=int, nargs='?', const=10, default=None, metavar='N', help='define distribution scale (default: 10)')
	parser.add_argument('--tune', dest='tune', type=int, nargs='?', const=10, default=None, metavar='N', help='tuning epochs (default: 10)', required=False)
	parser.add_argument('--setup', action='store_true', help='install dependencies')
	args = parser.parse_args()

	signal.signal(signal.SIGINT, signal_handler)
	title()

	if (not args.setup and not alive()):
		print("[!] Environment not setup.")
		parser.print_help()
		sys.exit(0)
	elif (args.setup): 
		if (not setup()): 
			print("[!] Setup could not complete.")
			sys.exit(0)
	print("[+] Environment ready.")

	# allocate distribution from args
	sets = []
	sets.append(["training", 0])
	sets.append(["testing", 0])

	# check for existing data
	create = False
	for dist in sets:
		prev = os.path.join(DIST, dist[0])
		try: open(prev)
		except IOError: open(prev, 'w+').close
		dist[1] = sum(1 for line in open(os.path.join(DIST, dist[0])))

	dsize = sets[0][1] + sets[1][1]
	if (args.dist):
		sets[0][1] = int(math.ceil((float(args.dist)/100)*75))
		sets[1][1] = int(math.ceil((float(args.dist)/100)*25))
		if (dsize>0):
			choice = raw_input("[!] Data found, recreate? (y/N)\n")
			if ("y" in choice or dsize==0):
				create = True
				sys.stdout.write("\033[F")
			else:
				sys.stdout.write("\033[F")
				print("[+] Skipping.")
		else: create = True
	elif (dsize==0):
		print("[!] No data found, specify new distribution scale (i.e. -d 50).\n")
		parser.print_help()
		sys.exit(0)

	os.system('stty -echo; setterm -cursor off;')	# disable user input

	# no data / overwrite == draw
	if (create):
		print("[!] Caution: This process can incur significant overhead.\n")

		for dist in sets:
			# create directories for code and executables
			if (os.path.isdir(os.path.join(HEAD, dist[0])) == True):
				shutil.rmtree(os.path.join(HEAD, dist[0]))
				os.mkdir(os.path.join(HEAD, dist[0]))
			# clear any previous distributions	
			open(os.path.join(DIST, dist[0]), 'w').close()

		# read preset labels
		with open(LABELS, 'rb') as labels:
			reader = csv.reader(labels)
			classes = list(reader)
			for dist in sets:
				samples = []
				instances = []
				for l, c in enumerate(classes):
					gen = Generate(int(dist[1]), c, dist[0])
					makefiles, obfuscations = gen.create()
					samples.append([l, makefiles])
					instances.append([c, obfuscations])

				# list samples and selected obfuscations
				print("[+] Generated: " + str(len(samples)*int(dist[1])))
				for i in instances:
					print("\t[-] " + '%10s' % ', '.join(i[0]) + " -- " + ', '.join("%s=%r" % (key,val) for (key,val) in i[1].iteritems()))
				print("")
				try:
					run_analysis(samples, os.path.join(DIST, dist[0]))
				except ZeroDivisionError:
					print("[!] Incorrect distribution scale.")

	# pass distribution to model
	train_model(sets, args.tune)

	# cleanup
	os.system('stty echo; setterm -cursor on;')
	if os.path.isdir("tmp"): shutil.rmtree("tmp")

if __name__ == "__main__":
	main(sys.argv)

