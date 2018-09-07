#!/usr/bin/env python

'''
	CryptoKnight
	@gregorydhill
'''

import os, sys, csv, datetime, shutil, subprocess, signal, threading, itertools, time, argparse
from src.setup import alive, setup
from src.title import title

BASE = os.path.dirname(os.path.realpath(__file__))
PIN = os.path.join(BASE, "pin_tool")
SAMPLE = os.path.join(BASE, "data", "sample")

def signal_handler(signal, frame):
	print('\n\n[!] Killing process and exiting.')
	os.system('stty echo; setterm -cursor on;')
	sys.exit(0)

def main(argv):
	parser = argparse.ArgumentParser(description='CryptoKnight - Assessment')
	parser.add_argument('-p', '--predict', type=str, metavar='args', \
		help='specify arguments as string literal')
	parser.add_argument('-e', '--evaluate', type=str, metavar='distribution', \
		help='specify set to evaluate')
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
	if (not args.predict and not args.evaluate):
		parser.print_help()
		sys.exit(0)

	arguments = args.evaluate if args.evaluate else args.predict.split(" ")[0]

	if not os.path.isfile(arguments):
		print("[!] " + str(arguments) + " is not a file.")
		sys.exit(0)

	# prevent user input during analysis
	os.system('stty -echo; setterm -cursor off;')

	# evaluate model on specified set
	if args.evaluate:
		print("[!] Evaluating: " + str(args.evaluate) + "\n")
		print("[+] x = predicted, y = actual")
		os.system("python ./src/Model/dcnn.py --evaluate " + args.evaluate)
		os.system('stty echo; setterm -cursor on;')
		sys.exit(0)

	arguments = args.predict.split(" ")		# collect exe specific args 
	open(SAMPLE, 'w').close()			# clean previous sample



	begin = False
	def animate():
		for c in itertools.cycle(['|', '/', '-', '\\']):
			if begin: break
			sys.stdout.write('\r[*] Tracing ' + c)
			sys.stdout.flush()
			time.sleep(0.2)

	# start timing
	start = datetime.datetime.now()
	print("[+] Start Time: " + str(start) + "\n")

	try:
		# trace animation
		t = threading.Thread(target=animate)
		t.daemon = True
		t.start()
	except (KeyboardInterrupt, SystemExit):
		begin = True
		print("\n[!] Exiting.")
		sys.exit(0)

	# run pintool in killable subprocess
	cmd = [PIN + "/pin", "-t", PIN + \
		"/source/tools/CryptoKnight/obj-intel64/CryptoTrace.so", \
		"-v", "3", "-o", SAMPLE, "--"]
	cmd.extend(arguments)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	
	try:
		while True:
			# read from stdout
			line = p.stdout.readline()
			if line:
				if not begin: 
					print("\r[+] Execution finished.\n")
					print("[=====]\n")
				begin = True
			else: break
			print(line),
	except KeyboardInterrupt:
		try:
			# user defined closure, terminate subprocess
			p.terminate()
		except OSError:
			pass
	p.wait()

	# collect timing information
	end = datetime.datetime.now()
	print("\n[+] End Time: " + str(end))
	mins, secs = divmod(((end-start).total_seconds()), 60)
	hours, mins = divmod(mins, 60)
	total = '%02d:%02d:%02d' % (hours, mins, secs)
	print("[+] Analysis Time: " + total)

	# evaluate model with custom sample
	os.system("python ./src/Model/dcnn.py --predict " + SAMPLE)

	# cleanup
	os.system('stty echo; setterm -cursor on;')
	if os.path.isdir("tmp"): shutil.rmtree("tmp")

if __name__ == "__main__":
	main(sys.argv)
