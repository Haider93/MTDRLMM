#include "../../include/data/basic.h"
#include "../../include/utilities/time.h"

#include <iostream>

using namespace std;
using namespace data::basic;

MarketDepth::MarketDepth():
    data::MarketDepth(),
    csv_()
{}

MarketDepth::MarketDepth(string file_path):
    MarketDepth()
{
    LoadCSV(file_path);
}

string file_path;//contain lob file path for _LoadAllRowUntil(rows)
void MarketDepth::LoadCSV(string path)
{
    Reset();

    csv_.openFile(path);
    csv_.skip(1);        // Ignore header
	
	file_path = path;
    LoadNext();
}


bool MarketDepth::_LoadAllRowUntil(int n) {
	long csv_pos = csv_.tellPosition();

	CSV csv_next_n_recs(file_path);
	//csv_next_n_recs.openFile(file_path);
	csv_next_n_recs.skip(1);

	//std::auto_ptr<CSV> csv_next_n_recs(new CSV());//smart pointer
	//csv_next_n_recs.openFile(file_path);
	csv_next_n_recs.setPosition(csv_pos);
	//csv_next_n_recs.setPosition(csv_next_n_recs.tellPrevPosition());//without prev position it runs ahead record_next i.e. 2 position ahead of curr record
	//csv_next_n_recs.setPosition(csv_.tellPrevPosition());
	while (n-- > 0) {
		if (row_1.size() != 22) {
			if (not csv_next_n_recs.hasData())
				return false;
			//csv_next_n_recs.next(row_1);//eating all memory
			csv_.next(row_1);
			if (row_1.size() != 22)
				return false;
			record_for_next_n_records.date = stoi(row_1.at(DATE));
			record_for_next_n_records.time = string_to_time(row_1.at(TIME));
			for (int i = 0; i < 5; i++) {
				double nap = stof(row_1.at(AP1 + i)),
					nbp = stof(row_1.at(BP1 + i));
				if (nap <= 0.0 or nbp <= 0.0) {
					row_1.clear();
					return false;
				}
				record_for_next_n_records.ask_prices[i] = nap;
				record_for_next_n_records.ask_volumes[i] = stol(row_1.at(AV1 + i));
				record_for_next_n_records.bid_prices[i] = nbp;
				record_for_next_n_records.bid_volumes[i] = stol(row_1.at(BV1 + i));
			}
			row_1.clear();
			next_n_records.push_back(record_for_next_n_records);
			record_for_next_n_records.clear();
		}
	}
	csv_next_n_recs.closeFile();
	return true;
}

bool MarketDepth::_LoadRow()
{
    while (row_.size() != 22) {
        if (not csv_.hasData())
            return false;

        csv_.next(row_);
		
    }

    if (row_.size() == 22)
        return true;
    else
        return false;
}
int countt = 0;
bool MarketDepth::_ParseRow()
{
    record_next.date = stoi(row_.at(DATE));
    record_next.time = string_to_time(row_.at(TIME));
	//cout << "Record " << countt++ << endl;
    for (int i = 0; i < 5; i++) {
        double nap = stof(row_.at(AP1 + i)),
               nbp = stof(row_.at(BP1 + i));

        if (nap <= 0.0 or nbp <= 0.0) {
            row_.clear();
            return false;
        }

		//cout << row_.at(1) << endl;
        record_next.ask_prices[i] = nap;
        record_next.ask_volumes[i] = stol(row_.at(AV1 + i));

        record_next.bid_prices[i] = nbp;
        record_next.bid_volumes[i] = stol(row_.at(BV1 + i));

		//cout << record_next.ask_prices[i] << ",";
		
    }
	//cout << endl;
    row_.clear();
    return true;
}

bool MarketDepth::_LoadNext()
{
    if (not csv_.isOpen())
        return false;

    while (true) {
        if (not _LoadRow())
            break;

        if (_ParseRow())
            return true;
    }

    return false;
}

long MarketDepth::_TimeLookAhead()
{
    if (not _LoadRow())
        return -1;
    else
        return string_to_time(row_.at(TIME));
}

void MarketDepth::Reset()
{
    Streamer::Reset();

    csv_.closeFile();
    row_.clear();
}

void MarketDepth::SkipN(long n)
{
    csv_.skip(n);
}

// ------------------------------------------------------------------

TimeAndSales::TimeAndSales():
    data::TimeAndSales(),

    csv_()
{}

TimeAndSales::TimeAndSales(string file_path):
    data::TimeAndSales()
{
    LoadCSV(file_path);
}

void TimeAndSales::LoadCSV(string path)
{
    Reset();

    csv_.openFile(path);
    csv_.skip(1);

    LoadNext();
}

bool TimeAndSales::_LoadAllRowUntil(int n) {
	return true;
}

bool TimeAndSales::_LoadRow()
{
    while (row_.size() != 4) {
        if (not csv_.hasData())
            return false;

        csv_.next(row_);
		//cout << "Time and sales record row : " << row_[0] << row_[1] << endl;
    }

    if (row_.size() == 4)
        return true;
    else
        return false;
}

bool TimeAndSales::_ParseRow()
{
    record_next.date = stoi(row_.at(DATE));
    record_next.time = string_to_time(row_.at(TIME));

	//cout << "Time and sles Time: " << row_.at(TIME) << endl;

    double price = stof(row_.at(PRICE));
    long size = stol(row_.at(SIZE));

    if (price > 0.0 and size > 0)
        record_next.transactions[price] += size;

    row_.clear();

	//cout << "Trade price:" << price << endl;

    return true;
}

bool TimeAndSales::_LoadNext()
{
    if (not csv_.isOpen() or not _LoadRow())
        return false;

    // Parse the row_
    long la = 0;
    long target = _TimeLookAhead();
    do {
        if (la == -1 or (not _ParseRow()))
            return false;

        la = _TimeLookAhead();

    } while (la <= target);

    return true;
}

long TimeAndSales::_TimeLookAhead()
{
    if (not _LoadRow())
        return -1;
    else
        return string_to_time(row_.at(TIME));
}

void TimeAndSales::Reset()
{
    Streamer::Reset();

    csv_.closeFile();
    row_.clear();
}

void TimeAndSales::SkipN(long n)
{
    csv_.skip(n);
}
