#include "../../include/environment/intraday.h"

#include "../../include/data/records.h"
#include "../../include/market/book.h"
#include "../../include/market/market.h"
#include "../../include/market/measures.h"
#include "../../include/utilities/time.h"
#include "../../include/utilities/maths.h"
#include "../../include/utilities/comparison.h"

#include <map>
#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <Python.h>


using namespace environment;
using namespace market::measure;

template<class T1, class T2>
const map<string, Variable> Intraday<T1, T2>::v_to_i = {
    {"pos", Variable::pos},
    {"spd", Variable::spd},
    {"mpm", Variable::mpm},
    {"imb", Variable::imb},
    {"svl", Variable::svl},
    {"vol", Variable::vol},
    {"rsi", Variable::rsi},
    {"vwap", Variable::vwap},
    {"a_dist", Variable::a_dist},
    {"a_queue", Variable::a_queue},
    {"b_dist", Variable::b_dist},
    {"b_queue", Variable::b_queue},
    {"last_action", Variable::last_action},
	{"m_sentiment", Variable::m_sentiment}
};

//predictor func
template<class T1, class T2>
float Intraday<T1, T2>::predictor(deque<double> input_seq, int sliding_window_size) {
	//Py_Initialize();

	////int sliding_window_size = 10;

	float prediction_result=0.0;

	//// Create some Python objects that will later be assigned values.

	//PyObject *pNamee, *pModulee, *pDictt, *pFuncc, *pArgss, *pValuee, *pListt = NULL, *pResultt = NULL;

	//// Convert the file name to a Python string.

	//string mod_name = "load_rf_regressor";
	//pName = PyUnicode_FromString("load_rf_regressor");

	//// Import the file as a Python module.

	//pModule = PyImport_Import(pName);

	//// Create a dictionary for the contents of the module.

	//pDict = PyModule_GetDict(pModule);

	//// Get the add method from the dictionary.

	//pFunc = PyDict_GetItemString(pDict, "model");
	//pFunc = PyObject_GetAttrString(pModule, "model");

	//// Create a Python tuple to hold the arguments to the method.

	//pArgs = PyTuple_New(1);

	// Convert 2 to a Python integer.

	//pValue = PyFloat_FromDouble(4.5);
	
		//python code goes here.

		deque<double> deq;
		deq = input_seq;
		double* array = new double[sliding_window_size];
		for (int i = 0; i < sliding_window_size; i++) {
			array[i] = deq.back();
			deq.pop_back();
		}

		pList = PyList_New(10);
		for (size_t i = 0; i != sliding_window_size; ++i) {
			PyList_SET_ITEM(pList, i, PyFloat_FromDouble(array[i]));
		}

		//// Set the Python int as the first and second arguments to the method.

		PyTuple_SetItem(pArgs, 0, pList);

		////PyTuple_SetItem(pArgs, 0, pValue);

		////PyTuple_SetItem(pArgs, 1, pValue);

		//// Call the function with the arguments.

		pResult = PyObject_CallObject(pFunc, pArgs);

		//// Print a message if calling the method failed.

		if (pResult == NULL)

			cout << "Calling the add method failed.\n";

		// Convert the result to a long from a Python object.

		prediction_result = PyFloat_AsDouble(pResult);


	/*predictor ends*/
	//Py_XDECREF(pName);
	//Py_XDECREF(pModule);
	////Py_XDECREF(pDictt);
	//Py_XDECREF(pFunc);
	//Py_XDECREF(pArgs);
	////Py_XDECREF(pValue);
	//Py_XDECREF(pList);
	//Py_XDECREF(pResult);
	//Py_Finalize();
	return prediction_result;
}



template<class T1, class T2>
Intraday<T1, T2>::Intraday(Config& c):
    Base(c),

    market_depth(),
    time_and_sales(),

    state_vars()
{

	/*string config_path = "C:\\Users\\SERG\\Desktop\\test.yaml";
	ifstream ifs(config_path, ifstream::in);
	YAML::Node c = YAML::Load(ifs);
	cout << c["learning"]["algorithm"] << endl <<
		c["data"]["symbols"] << " " <<
		c["debug"]["inspect_books"] << endl;*/
	/*ofstream temp_file("pred.csv", ofstream::out);
	temp_file << "Actual" + std::to_string(',') + "Predicted" + "\n";*/
	//Py_Initialize();
	//pName = PyUnicode_FromString("load_rf_regressor");

	//// Import the file as a Python module.

	//pModule = PyImport_Import(pName);

	//// Create a dictionary for the contents of the module.

	//pDict = PyModule_GetDict(pModule);

	//// Get the add method from the dictionary.

	//pFunc = PyDict_GetItemString(pDict, "model");
	//pFunc = PyObject_GetAttrString(pModule, "model");

	//// Create a Python tuple to hold the arguments to the method.

	//pArgs = PyTuple_New(1);



    static_assert(is_base_of<data::MarketDepth, T1>::value,
                  "T1 is not a subclass of data::MarketDepth");
    static_assert(is_base_of<data::TimeAndSales, T2>::value,
                  "T2 is not a subclass of data::TimeAndSales");

    auto v = c["state"]["variables"].as<list<string>>();
    for (auto it = v.begin(); it != v.end(); ++it) {
        try {
            state_vars.push_back(v_to_i.at(*it));

        } catch (const out_of_range& oor) {
            cout << "Unknown state variable: " << *it << "." << endl;

            throw;
        }
    }

    if (c["market"]["target_price"]["type"].as<string>() == "book") {
        l2p_ = [this](int al, int bl) {
            return std::make_tuple(
                market->ToPrice(market->ToTicks(ask_book_.price(0)) + al),
                market->ToPrice(market->ToTicks(bid_book_.price(0)) - bl)
            );
        };

    } else {
        l2p_ = [this](int al, int bl) {
            long double tp = target_price_->get(),
                   half_spd = max(0.0, spread_window.mean() / 2.0);

			long double ftp = next_state_target_price > 0.0 ? next_state_target_price : 0.0;

			//temp_file << std::to_string(tp) + std::to_string(',') + std::to_string(ftp) + "\n";

			//cout << "Target price :::" << tp << " Next target price  :::" << ftp << endl;

			//cout << " Ask order as per current target price :::" << market->ToPrice(market->ToTicks(tp + al * half_spd)) << " Bid order as per current target price :::" << market->ToPrice(market->ToTicks(tp - al * half_spd)) << endl;
			/*if (ftp > 0.0)
			{
				float price = (tp * 0.5) + (ftp * 0.5);
				tp = price;
			}*/

			//cout << " Average of tp and ftp :::" << tp << endl;
			//cout << " Ask order as per average price :::" << market->ToPrice(market->ToTicks(tp + al * half_spd)) << " Bid order as per avearge price :::" << market->ToPrice(market->ToTicks(tp - al * half_spd)) << endl;


            return std::make_tuple(
                market->ToPrice(market->ToTicks(tp + al*half_spd)), //ask quote of market maker (ref_price + thetaA*spread)
                market->ToPrice(market->ToTicks(tp - bl*half_spd)) //bid quote of market maker (ref_price - thetaB*spread)
            );
        };
    }
}

template<class T1, class T2>
Intraday<T1, T2>::Intraday(Config& c,
                           string symbol,
                           string md_path,
                           string tas_path):
    Intraday<T1, T2>(c)
{
    LoadData(symbol, md_path, tas_path);
}

template<class T1, class T2>
Intraday<T1, T2>::~Intraday()
{
    if (market != nullptr)
        delete market;
	//Base::~Base();
	//Py_XDECREF(pName);
	//Py_XDECREF(pModule);
	//Py_XDECREF(pDict);
	//Py_XDECREF(pFunc);
	//Py_XDECREF(pArgs);
	////Py_XDECREF(pValue);
	//Py_XDECREF(pList);
	//Py_XDECREF(pResult);
	//Py_Finalize();

}

template<class T1, class T2>
bool Intraday<T1, T2>::Initialise()
{
    bool stat = Base::Initialise();

    last_date = 0;
	market->set_date(0);
	market->set_time(0L);

    // Keep loading data until we are ready:
    while (not market->IsOpen())
        if (not UpdateBookProfiles())
            return false;

    time_and_sales.SkipUntil(market->date(), market->time());

    // Update all features with in-market data:
    while (not (f_ask_transactions.full() and
                f_bid_transactions.full() and
                f_vwap_numer.full() and
                f_vwap_denom.full() and
                f_volatility.full() and
                f_midprice.full() and
                target_price_->ready() and
                spread_window.full()))
        if (not NextState())
            return false;

    time_and_sales.SkipUntil(market->date(), market->time());

    ref_time = market->time();
    init_date = market->date();

    _place_orders(1, 1);

    return stat;
}

template<class T1, class T2>
void Intraday<T1, T2>::LoadData(string ticker, string md_path, string tas_path)
{
    market_depth.LoadCSV(md_path);
    time_and_sales.LoadCSV(tas_path);

    std::string symbol = ticker.substr(0, ticker.find_first_of('.'));
    std::string venue  = ticker.substr(ticker.find_first_of('.') + 1);

    market = market::Market::make_market(symbol, venue);
}

template<class T1, class T2>
bool Intraday<T1, T2>::isTerminal()
{
    return (not market->IsOpen()) or
        ((last_date != 0) and (market->date() != last_date));
}

template<class T1, class T2>
string Intraday<T1, T2>::getEpisodeId()
{ return to_string(init_date); }

template<class T1, class T2>
void Intraday<T1, T2>::_place_orders(int al, int bl, bool replace)
{
    ask_level = al;
    bid_level = bl;

	//sending negative price value for tick conversion...breaking code execution here
    std::tie(ask_quote, bid_quote) = l2p_(al, bl);

    risk_manager_.PlaceOrder(market::Side::ask, ask_quote, ORDER_SIZE, replace);
    risk_manager_.PlaceOrder(market::Side::bid, bid_quote, ORDER_SIZE, replace);
}

ofstream outfile1;

template<class T1, class T2>
void Intraday<T1, T2>::DoAction(int action, float regime)
{
    ref_time = (long) ceil(market->time() + latency_->sample());

	/*outfile1.open("C:\\Users\\Abbas\\Desktop\\abbas\\rl_markets\\temporary\\actions.txt", ios::out | ios::app);
	outfile1 << to_string(action) << endl;
	outfile1.close();*/
	int reg;
	if (regime >= 50.0)
		reg = 0;//extreme bearish
	else if (regime < 0.5 && regime >= 0.0)
		reg = 1;//extreme bullish
	else if (regime <= 10.0 && regime >= 0.5)
		reg = 2;//bullish
	else if (regime > 20.0 && regime < 50.0)
		reg = 3;//bearish
	else if (regime > 10.0 && regime <= 20.0)
		reg = 4;//neutral
	//outfile1 << regime << endl;
	//outfile1.close();


	//regime tells extreme bearishness or bullishness, use it fot that

	/*switch (reg) {
		case 0:
			_place_orders(0, 3);
			break;
		case 1:
			_place_orders(3, 0);
			break;
		case 2:
			_place_orders(1, 0);
			break;
		case 3:
			_place_orders(0, 1);
			break;
		case 4:
			_place_orders(1, 1);
			break;
	}*/

     //Do the action
    switch (action) {
        case 0:
            _place_orders(1, 1);
            break;

        case 1:
            ClearInventory();
            _place_orders(ask_level, bid_level);

            break;

        case 2:
            _place_orders(2, 2);
            break;

        case 3:
            _place_orders(3, 3);
            break;

        case 4:
            _place_orders(0, 2);
            break;

        case 5:
            _place_orders(2, 0);
            break;

        case 6:
            _place_orders(1, 4);
            break;

        case 7:
            _place_orders(4, 1);
            break;

        case 8:
            _place_orders(5, 5);
            break;
    }
}


// Read a bid or ask row from the data source
// Assumes that such a row is currently in the data source
template<class T1, class T2>
bool Intraday<T1, T2>::NextState()
{
    int target_date = market_depth.NextDate();
    long target_time = market_depth.NextTime();

    if (not time_and_sales.LoadUntil(target_date, target_time))
        return false;

    const data::TimeAndSalesRecord& rec_ts = time_and_sales.Record();

    double mp = midprice(ask_book_, bid_book_);
    auto au = ask_book_.ApplyTransactions(rec_ts.transactions, mp),
         bu = bid_book_.ApplyTransactions(rec_ts.transactions, mp);

	//target_price_->update(ask_book_, bid_book_);//target price should get updated before updating LOB curr row

    if (not UpdateBookProfiles(rec_ts.transactions))
        return false;

    auto adverse_selection =
        market::BookUtils::HandleAdverseSelection(ask_book_, bid_book_);

    pnl_step += get<1>(au) + get<1>(bu) + get<1>(adverse_selection);

	if(pnl_step>0)
		cout << "PnL ::" << pnl_step << endl;

    lo_vol_step += get<0>(bu) - get<0>(au) + abs(get<0>(adverse_selection));

	double sum = get<2>(au) + get<2>(bu) + get<2>(adverse_selection);
	if (sum < 0.0)
		int i = 0;

    episode_stats.pnl += get<2>(au) + get<2>(bu) + get<2>(adverse_selection);

	//experiment_stats.pnl += get<2>(au) + get<2>(bu) + get<2>(adverse_selection);

    // Update our position:
    risk_manager_.Update(get<0>(bu) + get<0>(au) + get<0>(adverse_selection));

    long mpt = market->ToTicks(midprice(ask_book_, bid_book_));
    double mpm = midprice_move(ask_book_, bid_book_),
           sp = spread(ask_book_, bid_book_);

    f_midprice.push(mpt);
    f_volatility.push(mpt);
    f_vwap_numer.push(ask_book_.observed_value() + bid_book_.observed_value());
    f_vwap_denom.push(ask_book_.observed_volume() + bid_book_.observed_volume());

    spread_window.push(max(0.0, sp));
    target_price_->update(ask_book_, bid_book_);

    return_ups.push(max(0.0, mpm));
    return_downs.push(abs(min(0.0, mpm)));

    f_ask_transactions.push(ask_book_.observed_volume());
    f_bid_transactions.push(bid_book_.observed_volume());

    return true;
}



template<class T1, class T2>
bool Intraday<T1, T2>::UpdateBookProfiles(
    const std::map<double, long, FloatComparator<>>& transactions)
{
    ask_book_.StashState();
    bid_book_.StashState();

	float prediction_result;

    while (true) {
        if (not market_depth.LoadNext()) return false;

        const data::MarketDepthRecord& rec_md = market_depth.Record();

		/*
		store midprice values as input to python predictor
		*/
		//auto recs = market_depth.Record();
		//input_price_predictor.push_back(midprice(rec_md.ask_prices.at(0), rec_md.bid_prices.at(0)));

        last_date = market->date();
        market->set_date(rec_md.date);
        market->set_time(rec_md.time);

        ask_book_.ApplyChanges(rec_md.ask_prices,
                               rec_md.ask_volumes,
                               transactions);

        bid_book_.ApplyChanges(rec_md.bid_prices,
                               rec_md.bid_volumes,
                               transactions);

        if (not market_depth.WillTimeChange()) continue;

        // Use a try as a bit of a shortcut. If the books haven't seen
        // enough (2) states to have a midprice move then it will clearly
        // throw an exception when we try to access it. So, instead, we just
        // use a try and start again if we couldn't.
        try {
			if (market::BookUtils::IsValidState(ask_book_, bid_book_)) {
				input_price_predictor.push_back(midprice(rec_md.ask_prices.at(0), rec_md.bid_prices.at(0)));
				break;
			}
                
        } catch (runtime_error &e) {
            continue;
        }
    }

	//future state asset prices in considerations
		//retreive next n records ahead of record_curr
	//try {
	//	auto next_n_recs = market_depth.LoadAllUntil(5);
	//	//retrieving next 1 state midprice
	//	/*outfile1.open("C:\\Users\\Abbas\\Desktop\\abbas\\rl_markets\\temporary\\states_list.txt", ios::out | ios::app);
	//	outfile1 << "Next price ::" << to_string(midprice(next_n_recs.at(0).ask_prices.at(0), next_n_recs.at(0).bid_prices.at(0))) << ",Current price ::" << to_string(target_price_->get()) << endl;
	//	outfile1.close();*/

	//	if (next_n_recs.size() != 0){
	//		next_state_target_price = midprice(next_n_recs.at(0).ask_prices.at(0), next_n_recs.at(0).bid_prices.at(0));
	//	}

	//}
	//catch (exception e) {
	//	cout << "Exception occccurred" << endl;
	//}


	

	//auto rec = market_depth.Next_Record();
	//next_state_target_price = midprice(rec.ask_prices.at(0), rec.bid_prices.at(0));
	/*if (input_price_predictor.size() == 10)
	{
		prediction_result = predictor(input_price_predictor, 10);
		input_price_predictor.pop_front();
	}
	next_state_target_price = prediction_result;*/
	//cout << "Prediction result :" << prediction_result << endl;

	//Py_Finalize();
    return true;
}

template<class T1, class T2>
double Intraday<T1, T2>::getVariable(Variable v)
{
    switch (v) {

		case Variable::m_sentiment:
			//return target_price_.top();
			//return target_price_->avg();
			//return (rand() % 9999999999 + 999) / 1.0;
			//return (rand() % 11) / 1.0;
			return target_price_->get();


        case Variable::pos:
            // Generalise -> ORDER_SIZE (default: 1)
            return double(risk_manager_.exposure()) / ORDER_SIZE;

        case Variable::spd:
            // Generalise -> 1 tick
            return ulb((double)(market->ToTicks(ask_book_.price(0)) -
                                market->ToTicks(bid_book_.price(0))),
                       0.0, 20.0);

        case Variable::mpm:
            // Generalise -> 1 tick
			//cout << "Inside MPM" << endl;
            return ulb(
                (double)(market->ToTicks(f_midprice.front()) -
                         market->ToTicks(f_midprice.back())),
                -10.0, 10.0
                );

        case Variable::imb: {
            double v_a = (double) ask_book_.total_volume(),
                   v_b = (double) bid_book_.total_volume();

            // Generalise -> 0.2
            return ((v_a + v_b) > 0 ? 5*(v_b - v_a) / (v_b + v_a) : 0.0);
        }

        case Variable::svl: {
            double q_a = (double) f_ask_transactions.sum(),
                   q_b = (double) f_bid_transactions.sum();

			//cout << "Ask transactions:" << q_a << " ,Bid transactions::" << q_b << endl;

            // Generalise -> 0.2
            return ((q_a + q_b) > 0 ? 5*(q_b - q_a) / (q_a + q_b) : 0.0);
        }

        case Variable::vol:
            return ulb(5.0*f_volatility.std(), 0.0, 10.0);

        case Variable::rsi: {
            double u = return_ups.mean(),
                   d = return_downs.mean();

            // Generalise -> 0.20
            return (u + d) != 0.0 ? 5.0 * (u - d) / (u + d) : 0.0;
        }

        case Variable::vwap: {
            double d = f_vwap_numer.sum() / f_vwap_denom.sum();

            return ulb(
                d / spread_window.mean(), -10.0, 10.0
            );
        }

        case Variable::a_dist:
            // Generalise -> 1 tick
            if (ask_book_.order_count() > 0)
                return ((double) market->ToTicks(ask_book_.best_open_order_price()) -
                        (double) market->ToTicks(ask_book_.price(0)));
            else
                return -100.0;

        case Variable::a_queue:
            // Generalise -> 10%
            if (ask_book_.order_count() > 0)
                return 10.0 * ask_book_.queue_progress();
            else
                return -1.0;

        case Variable::b_dist:
            // Generalise -> 1 tick
            if (bid_book_.order_count() > 0)
                return ((double) market->ToTicks(bid_book_.price(0)) -
                        (double) market->ToTicks(bid_book_.best_open_order_price()));
            else
                return -100.0;

        case Variable::b_queue:
            // Generalise -> 10%
            if (bid_book_.order_count() > 0)
                return 10.0 * bid_book_.queue_progress();
            else
                return -1.0;

        case Variable::last_action:
            return last_action;

        default:
            throw std::invalid_argument("Unknown state-var enum value: " +
                                        to_string((int) v) + ".");
    }
}

template<class T1, class T2>
void Intraday<T1, T2>::getState(std::vector<float>& out)
{
	int len = 0;
	for (auto it = state_vars.begin(); it != state_vars.end(); ++it) {
		out.push_back(getVariable(*it));
	}

	/*cout << "Inside getState() values of state vars::" << endl;
	for (int i = 0; i < out.size(); i++)
		cout << out.at(i) << " ";
	cout << endl;*/
}

template<class T1, class T2>
void Intraday<T1, T2>::printInfo(const int action)
{
    cout << "Date: " << market->date() << endl;
    cout << "Time: " << time_to_string(market->time()) << "ms" << endl;

    cout << "Time (TAS): " << time_to_string(time_and_sales.Record().time) << endl;
    cout << "Time (TAS_f): " << time_to_string(time_and_sales.NextTime()) << endl << endl;

    if (market->IsOpen())
        cout << "Market status: Open" << endl;
    else
        cout << "Market status: Closed" << endl;

    cout << endl;

    Base::printInfo(action);
}

template<class T1, class T2>
void Intraday<T1, T2>::LogProfit(int action, double pnl, double bandh)
{
    if (profit_logger != nullptr)
        profit_logger->info("{},{},{},{},{},{},{},{},{},{},{},{}",
                            market->date(),
                            market->time(),
                            action,
                            risk_manager_.exposure(),
                            midprice(ask_book_, bid_book_),
                            spread(ask_book_, bid_book_),
                            ask_quote, bid_quote,
                            ask_level, bid_level,
                            pnl, bandh);
}

template<class T1, class T2>
void Intraday<T1, T2>::LogTrade(char side, char type, double price,
                                long size, double pnl)
{
    if (trade_logger != nullptr)
        trade_logger->info("{},{},{},{},{},{},{},{}",
                           market->date(), market->time(),
                           risk_manager_.exposure(),
                           side, type, price, size, pnl);
}

// Template specialisations
template class environment::Intraday<data::basic::MarketDepth,
                                     data::basic::TimeAndSales>;
