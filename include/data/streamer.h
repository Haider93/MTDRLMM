#ifndef DATA_STREAMER_H
#define DATA_STREAMER_H

#include <map>
#include <string>
#include <vector>

#include "../data/records.h"
#include "../utilities/csv.h"

using namespace std;

namespace data
{

template<typename R>
class Streamer
{
    protected:
        R record_1;
        R record_2;
        R record_3;

        R& record_last;
        R& record_curr;
        R& record_next;
		vector<R> next_n_records;//holds next records ahead of record_curr
		R record_for_next_n_records;//used in _LoadAllRowUntil(no of rows)

        Streamer();

        void _RotateRefs();

        virtual bool _LoadNext() = 0;
        virtual long _TimeLookAhead() = 0;
		virtual bool _LoadAllRowUntil(int n) = 0;//loads n rows in next_10_records

    public:
        bool LoadNext();

        bool LoadUntil(const data::Record& rec);
        bool LoadUntil(int date, long time);

        bool SkipUntil(const data::Record& rec);
        bool SkipUntil(int date, long time = 0);

        const R& Record();
		const R& Next_Record();

		const vector<R> LoadAllUntil(int n);//calls _LoadAllRowUntil(no of rows)

        virtual void Reset();
        virtual void SkipN(long n = 1L) = 0;

        bool HasTimeChanged();
        bool WillTimeChange();

        int NextDate();
        long NextTime();
};

class MarketDepth: public Streamer<MarketDepthRecord> {};
class TimeAndSales: public Streamer<TimeAndSalesRecord> {};

}

#endif
