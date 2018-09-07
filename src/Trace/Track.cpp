#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <memory>
#include <unistd.h>
#include <iterator>
#include <vector>
#include <cstdio>
#include <stdlib.h>
#include <math.h> 
#include "pin.H"

#include "Track.hpp"

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

#define ATR 16

using namespace std;

Track::Track(vector<string> path, string ent_file)
{
	enabled = false;
	files = path;
	EntFile.open(ent_file.c_str());
}

void Track::activate(int opts)
{
	enabled = true;
	cout << endl;
}

void Track::addEvent(string event, bool imm = false)
{
	if (enabled & imm) cout << event << endl;
	else report.append(event + "\n");
}

void Track::write()
{
	if (enabled) cout << report;
}

/*
int Track::trackENT(vector<pair<string, double> > mem)
{
	long score = 0;
	double last_ent = 0;

	for (const auto& ent : mem) 
	{
		if (ent.second>last_ent) score+=1;
		else score-=1;
		last_ent = ent.second;
	}

	return std::labs(score);
}*/


int Track::trackENT(vector<pair<string, double> > mem)
{
	map<string, long> score;
	map<string, double> last_ent;

	for (const auto& ent : mem) 
	{
		if (ent.first=="") continue;
		if (score.find(ent.first) == score.end()) {
			score[ent.first]=0;
			last_ent[ent.first]=ent.second;
		}
		else {
			if (ent.second>last_ent[ent.first]) score[ent.first]+=1;
			else score[ent.first]-=1;
			last_ent[ent.first] = ent.second;
		}
	}
	
	int final_score = 0;

	for (const auto& s : score) {
		final_score+=std::labs(s.second);
	}

	EntFile << "(" << count << "," << final_score << ")";
	count++;
	return final_score;
}

int Track::createWord(BLOCK bbl) 
{
	vector<int> word(ATR, 0);
	int ratio = 0;
	int score = 0;
	int trace = 0;

	// define number of embeddings in preamble
	for (const auto& ins : bbl.body)
	{	
		switch(ins.second) {
			case XED_ICLASS_CMP     : { word.at(0)++; ratio++; break; }
			case XED_ICLASS_MOV     : { word.at(1)++; ratio++; break; }
			case XED_ICLASS_TEST    : { word.at(2)++; ratio++; break; }
			case XED_ICLASS_LEA     : { word.at(3)++; ratio++; break; }
			case XED_ICLASS_AND     : { word.at(4)++; ratio++; break; }
			case XED_ICLASS_OR      : { word.at(5)++; ratio++; break; }
			case XED_ICLASS_XOR     : { word.at(6)++; ratio++; break; }
			case XED_ICLASS_PXOR    : { word.at(7)++; ratio++; break; }
			case XED_ICLASS_ADD     : { word.at(8)++; ratio++; break; }
			case XED_ICLASS_SUB     : { word.at(9)++; ratio++; break; }
			case XED_ICLASS_INC     : { word.at(10)++; ratio++; break; }
			case XED_ICLASS_DEC     : { word.at(11)++; ratio++; break; }
			case XED_ICLASS_SHR    	: { word.at(12)++; ratio++; break; }
			case XED_ICLASS_SHL     : { word.at(13)++; ratio++; break; }
			case XED_ICLASS_SAR     : { word.at(14)++; ratio++; break; }
			case XED_ICLASS_NOT		: { word.at(15)++; ratio++; break; }
		}
	}

	score = trackENT(bbl.entropy);
	if (ratio>=round((bbl.body.size()/100)*55)&&ratio!=0) trace += 1;
	if (bbl.loop) trace += 2;
	if (score!=0)  trace += 4;

	for (int i=0; i<ATR; i++)
	{
		if (trace & 2)
		{
			word.at(i) *= bbl.iterations;
			word.at(i) += bbl.iterations;
		}

		if (trace & 4) 
			word.at(i) *= score;

		word.at(i) = abs(word.at(i));
	}
	
	if (trace & 1) sentence.push_back(word);

	return trace;
}

string Track::hex_to_string(const string& input)
{
	static const char* const lut = "0123456789ABCDEF";
	size_t len = input.length();

	string output;
	output.reserve(len / 2);
	for (size_t i = 0; i < len; i += 2)
	{
		char a = input[i];
		const char* p = std::lower_bound(lut, lut + 16, a);

		char b = input[i + 1];
 		const char* q = std::lower_bound(lut, lut + 16, b);

		output.push_back(((p - lut) << 4) | (q - lut));
	}
	return output;
}

void Track::trackBBL() 
{
	addEvent("[=====]\n", true);

	map<ADDRINT, BLOCK> blocks;
	vector<ADDRINT> bbl_order;
	ADDRINT last_head;

	for (const auto& file : files) {

		ifstream _ifile(file.c_str(), ios::binary);

		if (_ifile.is_open()) {

			string addr, op, dis, reg, ent;
			bool head, tail;

			while (_ifile >> addr >> op >> dis >> head >> tail >> reg >> ent)
			{
				string regist = "";
				if (reg!="0") regist = hex_to_string(reg);
				ADDRINT ins = AddrintFromString(addr);
				OPCODE opcode = Uint64FromString(op);
				double entropy = atof(ent.c_str());				

				if (head) 
				{
					blocks[ins].iterations += 1;
					if (blocks[ins].iterations==1) {
						blocks[ins].disassemble = hex_to_string(dis);
						bbl_order.push_back(ins);
					}

					if (!blocks[ins].loop&&last_head==ins)
						blocks[ins].loop = true;

					last_head = ins;
				
				}
				else if (blocks[last_head].iterations==1) 
				{
					blocks[last_head].body.push_back(make_pair(ins, opcode));
				}

				blocks[last_head].entropy.push_back(make_pair(regist, entropy));
			}
		}
		_ifile.close();
		remove(file.c_str());
	}

	for (const auto& b : bbl_order)
	{
		if (blocks[b].iterations>1 && blocks[b].loop) 
		{
			t_loops++;
			t_it += blocks[b].iterations;
		}
		blocks[b].reason = createWord(blocks[b]);
	}

	addEvent("[+] Feature Blocks: " + SSTR(sentence.size()));
	addEvent("[+] Total Loop Count: " + SSTR(t_loops));
	addEvent("[+] Total Loop Iterations: " + SSTR(t_it));

	if (enabled) buildReport(blocks, bbl_order);
}

void Track::buildReport(map<ADDRINT, BLOCK> blocks, vector<ADDRINT> bbl_order)
{
	char temp[128];
	ofstream outfile;
	outfile.open("report.txt", ios::trunc);

	outfile << "Report Summary\nTotal Loop Count: " << t_loops
		<< "\nTotal Loop Iterations: " << t_it << endl << endl;

	ostringstream table;
	table << setw(60) << left << "BBL Head" << setw(20) << "Iterations" 
		<< setw(10) << "Interest" << endl << endl;

	for (const auto& b : bbl_order) 
	{
		if (blocks.find(b) != blocks.end() && blocks[b].iterations>1) 
		{
			if (blocks[b].reason != 0) {
				table << setw(60) << left << blocks[b].disassemble
					<< setw(20) << SSTR(blocks[b].iterations)
					<< setw(10) << SSTR(blocks[b].reason) << endl;
			}
			else {
				table << setw(60) << left << blocks[b].disassemble
					<< setw(20) << SSTR(blocks[b].iterations) << endl;
			}
		}
    	}

	outfile << table.str() << endl;

	outfile.close();
	string cwd = (getcwd(temp, 128) ? std::string(temp) : std::string(""));
	addEvent("[+] Full trace report: " + cwd + "/report.txt", true);
}

void Track::buildFeature(string data_set, string label)
{
	string feature;
	for (const auto& word : sentence) {
		for (const auto& attribute : word) {
			feature.append(SSTR(attribute) + ",");
		}
		feature = feature.substr(0, feature.size()-1);
		feature.append(":");
	}
	feature = feature.substr(0, feature.size()-1);
	feature.append("\n");

	//cout << "\n" << feature << endl;

	ofstream outfile;
	outfile.open(data_set.c_str(), ios_base::app);
	if (label!="") { 
		outfile << label << ";" << feature; 
	}
	else {
		outfile << feature; 
	}
	outfile.close();
}

