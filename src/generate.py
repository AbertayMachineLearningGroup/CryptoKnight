#!/usr/bin/env python

'''
	CryptoKnight
	@gregorydhill
'''

import os, random, sys, getopt, json, string, collections, csv, argparse
import numpy as np
from Crypto.PublicKey import RSA

BASE = os.path.dirname(os.path.realpath(__file__))
HEAD = os.path.join(BASE, "../data")
CONF = os.path.join(BASE, "../data/config")
FLAGS = os.path.join(CONF, "flags")
LABELS = os.path.join(CONF, "labels")
POOL = os.path.join(CONF, "pool")
FABRICATOR = os.path.join(CONF, "fab")
CERTS = os.path.join(CONF, "certs")

probability = 25
primitives = {}

with open(POOL, 'rb') as samples:
	cores = csv.reader(samples)
	for core in cores:
		primitives[core[0]] = core[1]

store = {	"symmetric" : "unsigned char ciphertext[128];\n\tint ciphertext_len, len;\n\t", \
			"asymmetric" : "unsigned char rivest[4098];\n\t", "hash" : "unsigned char digest[16];\n\t"}

symmetric = {	"aes": {"key" : ["__key__", 256], "iv" : ["__iv__", 128]}, \
				"blowfish": {"key" : ["__key__", 256], "iv" : ["__iv__[8]", 8]}, \
				"rc4": {"key" : ["__key__", 256], "iv" : ["__iv__", 128]}}

intvar = ["int"]
arithmetics = {"add" : "+", "sub" : "-", "mul" : "*", "div" : "/"}
logics = {"and" : "&", "or" : "|", "xor" : "^"}
shifts = {"left" : "<<", "right" : ">>"}

class Application:
	def __init__(self, base, crypto, label, weights, id):
		self.base = base
		self.crypto = crypto
		self.label = label
		self.id = id
		self.variables = []
		self.parameters = {}
		self.obfuscation = np.random.choice(['aggregate', 'split', 'normal'], p=weights)
		if len(crypto) is 1:
			if primitives.get(crypto[0]) == 'hash' or primitives.get(crypto[0]) == 'asymmetric':
				self.obfuscation = 'normal'

	def instruction(self, hashmap, reuse = False):
		key, value = random.choice(list(hashmap.items()))
		create = bool(random.getrandbits(1))
		if len(self.variables) < 3: create = True
		if len(self.variables) < 3 and reuse: return ""
		if reuse: create = False
		range1 = [1, 100]
		range2 = [1, 100]
		if key == "right" or key == "left":
			range1 = [100, 200]
			range2 = [1, 3]
		instruction = ""
		if create:
			variable = "op_" + key + "_" + str(len(self.variables)+1)
			instruction = str(random.choice(intvar)) + " " + variable + " = " + \
				str(random.randint(range1[0], range1[1])) + value + str(random.randint(range2[0], range2[1])) + ";\n\t"
			self.variables.append(variable)
		else:
			choice = random.getrandbits(3)
			if choice == 0:
				instruction = str(random.choice(self.variables)) + " = " + \
					str(random.choice(self.variables)) + value + str(random.choice(self.variables)) + ";\n\t"
			elif choice == 1:
				instruction = str(random.choice(self.variables)) + " = " + \
					str(random.choice(self.variables)) + value + str(random.randint(range2[0], range2[1])) + ";\n\t"
			elif choice == 2:
				instruction = str(random.choice(self.variables)) + " " + \
					value + "= " + str(random.choice(self.variables)) + ";\n\t"
			else: instruction = str(random.choice(self.variables)) + " " + value + "= " + str(random.randint(range2[0], range2[1])) + ";\n\t"
		return instruction

	def inc_dec(self):
		if len(self.variables) == 0: return ""
		ins = str(random.choice(self.variables))
		choice = bool(random.getrandbits(1))
		if choice: ins += "++;\n\t"
		else: ins += "--;\n\t"
		return ins

	def loop(self):
		if len(self.variables) == 0: return ""
		create = bool(random.getrandbits(1))
		if not create: return ""
		instruction = ""
		l = random.randint(1, probability)
		instruction = str("for(int i=0; i<" + str(l) + "; i++) {\n\t")
		for y in range(random.randint(1, probability)):
			m = random.choice([self.instruction, self.inc_dec])
			if m == self.instruction:
				ins = m(random.choice([arithmetics, logics, shifts]), True)
				if ins != None: instruction += "\t" + ins
			else: 
				ins = m()
				if ins != None: instruction += "\t" + ins
		instruction += "}\n\t"
		return instruction

	def noise(self, cfile):
		create = bool(random.getrandbits(1))
		if not create: return
		for x in range(random.randint(0, probability)):
			methods = [self.instruction, self.inc_dec, self.loop]
			random.shuffle(methods)
			for m in methods:
				if m == self.instruction:
					cfile.write(m(random.choice([arithmetics, logics, shifts])))
				else: 
					cfile.write(m())

	def combine(self, output, key, value):
		try: output[key].append(value)
		except KeyError: output[key] = [value]

	def parameter(self, param, crypto):
		tokens = symmetric.get(crypto)
		if tokens is None: return
		tokens = tokens.get(param)
		rand = ''.join([random.choice(string.ascii_uppercase) for i in range(probability)])
		k1 = tokens[0].replace("__" + param + "__", "*" + rand)
		k2 = tokens[0].replace("__" + param + "__", rand)
		k3 = tokens[0].replace("__" + param + "__", rand)
		var_type = "unsigned char "
		end = " = \"" + str(random.getrandbits(tokens[1])) + "\";\n\t"

		if self.obfuscation == "aggregate":
			k1 = var_type + k1 + ";\n"
			k2 = "X." + k2
			if "[" and "]" in k2: k2 = "*" + k2
			k2 = k2.split("[")[0] + end
			k3 = "X." + k3
			if "[" and "]" in k3: k3 = "(unsigned char *)" + k3.split("[")[0]
		else:
			k1 = 0
			if "[" and "]" not in k2: k2 = "*" + k2
			k2 = var_type + k2 + end

		if self.obfuscation == "split" and param == "key":
			k2 += "unsigned char " + rand + "1[256];\n\tunsigned char " + rand + "2[256];\n\tfor(int i=0; i<256; i++) {\n\t\t" + rand + \
				"1[i] = " + rand + "[i] & 0x000FFFFF;\n\t\t" + rand + "2[i] = " + rand + "[i] >> 20 & 0x00000FFF;\n\t}\n\t"

		self.combine(self.parameters, crypto, [k1, k2, k3])

	def review(self, key, value):
		line = value[0]
		if "__key_def__" in line:
			key_def = self.parameters.get(key)[0][2]
			if "[" and "]" in key_def:
				key_def = "*" + key_def.split("[")[0]
			if self.obfuscation == "split":
				line = "unsigned char " + key_def + "3[256];\n\tfor(int i=0; i<256; i++) " + \
					key_def + "3[i] = " + key_def + "2[i] << 20 | " + key_def + "1[i];\n\t" + line
			line = line.replace("__key_def__", key_def)
		if "__iv_def__" in line:
			iv_def = self.parameters.get(key)[1][2]
			if "[" and "]" in iv_def:
				if self.obfuscation == "aggregate":
					iv_def = "(unsigned char *)" + iv_def
				iv_def = iv_def.split("[")[0]
			line = line.replace("__iv_def__", iv_def)
		return line 

	def create(self):
		imports = []
		cipher = []
		encryption = collections.OrderedDict()
		public_key = ""

		for crypto in self.crypto:
			constructor = open(os.path.join(FABRICATOR, crypto + '.json'))
			base = constructor.read()
			constructor.close()
			fab = json.loads(base)
			imports.append(fab['import'])
			self.parameter("key", crypto)
			self.parameter("iv", crypto)
			self.combine(encryption, crypto, fab['encrypt'])
			cipher.append(store.get(primitives.get(crypto)))
			if primitives.get(crypto) == "asymmetric":
				certname = random.choice(os.listdir(CERTS))
				with open(os.path.join(CERTS, certname)) as cert:
					public_key += cert.read()

		app = []
		app.append(''.join(imports))

		if self.obfuscation == "aggregate":
			struct = "struct {\n"
			for key, value in self.parameters.iteritems():
				for item in value: struct += item[0]
			struct += "} X;\n"
			app.append(struct)

		app.append("\nint main (void)\n{\n\t")
		app.append(public_key)
		for key, value in self.parameters.iteritems():
			for item in value: app.append(item[1])
		app.append("unsigned char *plaintext = \"" + \
			str(''.join(random.choice(string.ascii_letters + string.digits) for i in range(64))) + "\";\n\t")
		for c in set(cipher): app.append(c)
		for key, value in encryption.iteritems():
			app.append("// " + key + " routines\n\t" + self.review(key, value))

		app.append("\n\treturn 0;\n}\n")

		cfile = open(os.path.join(self.base, str(self.id) + '.c'), 'w+')
		main = False
		for entry in app:
			if "main" in entry: main = True
			if "return" in entry: main = False
			cfile.write(entry)
			if main is True: self.noise(cfile)

		return self.obfuscation


class Generate:
	def __init__(self, number, crypto, SET):
		self.files = number
		self.flags = []
		self.crypto = crypto
		self.label = label = ''.join(crypto)

		for c in crypto:
			if not os.path.isfile(os.path.join(FABRICATOR, c + ".json")):	
				print("[!] Primitive \'" + c + "\' not found.")
				sys.exit(2)

		with open(FLAGS, 'r') as f:
			for flag in f:
				if flag.strip():
					switch = flag.split("#")[0].rstrip()
					if switch:
						self.flags.append(switch)

		# create set-type dir
		if (os.path.isdir(os.path.join(HEAD, SET)) == False):
			os.mkdir(os.path.join(HEAD, SET))

		self.base = os.path.join(HEAD, SET, self.label)

		# create sub-set crypto dir
		if (os.path.isdir(self.base) == False):
			os.mkdir(self.base)

		self.make_file = open(os.path.join(self.base, 'Makefile'), 'w+')
		self.make_file.write("all:\n")
	
	def add_rule(self, x):
		cflag = random.choice(self.flags)
		self.make_file.write("\tgcc -std=c11 " + cflag + " -o " + self.label + "_" + str(x) + ".exe" + " " + str(x) + ".c -lcrypto\n")

	def create(self):
		choices = ['aggregate', 'split', 'normal']
		selection = {'aggregate' : 0, 'split': 0, 'normal' : 0}
		start = 1/float(len(selection))
		weights = [start, start, start]
		for x in range(int(self.files)):
			app = Application(self.base, self.crypto, self.label, weights, x)
			obf = app.create()
			selection[obf]+=1
			for i, k in enumerate(choices):
				if weights[i] - 0.05 or weights[i] - 0.025 < 0: break
				if k == obf: weights[i] -= 0.05
				else: weights[i] += 0.025
			self.add_rule(x)

		self.make_file.write("\n")
		self.make_file.close()
		return (self.base, selection)

if __name__ == "__main__":
	basepath = os.path.dirname(os.path.realpath(__file__))
	parser = argparse.ArgumentParser(description='CryptoKnight')
	parser.add_argument('-c', '--certs', type=int, metavar='N', help='number of certs')
	parser.add_argument('-r', '--regen', action='append', nargs=3, metavar=('number','crypto','set'), help='define a regen set')
	args = parser.parse_args()

	if args.certs:
		for i in range(args.certs):
			sys.stdout.write("\r[+] Generated: " + str(i+1))
			sys.stdout.flush()
			private = RSA.generate(2048)
			public = private.publickey()
			key = public.exportKey().replace("\n", "\\n\"\\\n\t\"")
			public_key = ("char publicKey[]=\"" + key + "\";\n\tint padding = RSA_PKCS1_PADDING;\n\t")
			with open(os.path.join(basepath, CERTS, "cert_" + str(i)), 'w+') as cert:
				cert.write(str(public_key))
		print("")

	if args.regen:
		regen = args.regen
		for r in regen:
			try:
				components = r[1].split(',')
				gen = Generate(int(r[0]), components, r[2])
				gen.create()
				print("[+] Generated: " + str(r[0]) + " - \'" + ", ".join(components) + "\'")
			except ValueError:
				print("[!] Invalid argument specified.")




