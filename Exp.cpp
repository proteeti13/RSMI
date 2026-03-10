#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <boost/algorithm/string.hpp>
#include "utils/FileReader.h"
#include "indices/RSMI.h"
#include "utils/ExpRecorder.h"
#include "utils/Constants.h"
#include "utils/FileWriter.h"
#include "utils/util.h"
#include "utils/ModelTools.h"
#include <torch/torch.h>

#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

using namespace std;

#ifndef use_gpu
// #define use_gpu

long long cardinality   = 10000;
string distribution     = "triples";
int skewness            = 1;
string dataset_file     = "";
long long limit         = 1000000;   // max tuples to load (default 1M)
int num_queries         = 100000;    // lookup benchmark queries
int query_seed          = 42;        // random seed for reproducibility

string RSMI::model_path_root = "";

int main(int argc, char **argv)
{
    int c;
    static struct option long_options[] =
    {
        {"cardinality",  required_argument, NULL, 'c'},
        {"distribution", required_argument, NULL, 'd'},
        {"skewness",     required_argument, NULL, 's'},
        {"dataset",      required_argument, NULL, 'f'},
        {"limit",        required_argument, NULL, 'l'},
        {"queries",      required_argument, NULL, 'q'}
    };

    while (1)
    {
        int opt_index = 0;
        c = getopt_long(argc, argv, "c:d:s:f:l:q:", long_options, &opt_index);
        if (-1 == c) break;
        switch (c)
        {
            case 'c': cardinality  = atoll(optarg); break;
            case 'd': distribution = optarg;        break;
            case 's': skewness     = atoi(optarg);  break;
            case 'f': dataset_file = optarg;        break;
            case 'l': limit        = atoll(optarg); break;
            case 'q': num_queries  = atoi(optarg);  break;
        }
    }

    // ---- Resolve dataset filename ----
    string dataset_filename;
    if (!dataset_file.empty())
    {
        dataset_filename = dataset_file;
        size_t slash = dataset_file.rfind('/');
        string stem = (slash == string::npos) ? dataset_file : dataset_file.substr(slash + 1);
        size_t dot = stem.rfind('.');
        if (dot != string::npos) stem = stem.substr(0, dot);
        distribution = stem;
    }
    else
    {
        string size_tag = (cardinality <= 10000) ? "10k" : "50k";
        dataset_filename = Constants::DATASETS + "triples_" + size_tag + ".txt";
    }

    // ---- Load data ----
    FileReader filereader(dataset_filename, " ");
    vector<Point> all_points = filereader.get_points();

    if (all_points.empty())
    {
        cerr << "ERROR: no points loaded from " << dataset_filename << endl;
        return 1;
    }

    // Truncate to limit
    if ((long long)all_points.size() > limit)
        all_points.resize((size_t)limit);

    cardinality = (long long)all_points.size();
    cerr << "Loaded " << cardinality << " triples from " << dataset_filename << endl;

    // ---- Prepare model directory ----
    string model_root_path = Constants::TORCH_MODELS + distribution + "_" + to_string(cardinality);
    file_utils::check_dir(model_root_path);
    string model_path = model_root_path + "/";

    // ---- BUILD (silent) ----
    rsmi_verbose = false;

    ExpRecorder build_rec;
    build_rec.clean();                              // zero-initialize all counters
    build_rec.dataset_cardinality = cardinality;
    build_rec.distribution        = distribution;
    build_rec.skewness            = skewness;

    RSMI::model_path_root = model_path;
    RSMI *partition = new RSMI(0, Constants::MAX_WIDTH);
    partition->model_path = model_path;

    auto build_start  = chrono::high_resolution_clock::now();
    partition->build(build_rec, all_points);
    auto build_finish = chrono::high_resolution_clock::now();
    long long build_ns = chrono::duration_cast<chrono::nanoseconds>(build_finish - build_start).count();

    // Index size in bytes
    long long index_bytes =
        (Constants::DIM * Constants::HIDDEN_LAYER_WIDTH
         + Constants::HIDDEN_LAYER_WIDTH
         + Constants::HIDDEN_LAYER_WIDTH + 1)
        * Constants::EACH_DIM_LENGTH * build_rec.non_leaf_node_num
        + (Constants::DIM * Constants::PAGESIZE
           + Constants::PAGESIZE
           + Constants::DIM * 2)
        * Constants::EACH_DIM_LENGTH * build_rec.leaf_node_num;

    // ---- QUERY BENCHMARK ----
    // Sample num_queries points with fixed seed
    std::mt19937_64 bench_rng(query_seed);
    std::uniform_int_distribution<long long> bench_dist(0, cardinality - 1);

    vector<Point> queries;
    queries.reserve(num_queries);
    for (int i = 0; i < num_queries; i++)
        queries.push_back(all_points[bench_dist(bench_rng)]);

    // Time each query individually
    vector<long long> latencies;
    latencies.reserve(num_queries);
    double total_page_access = 0.0;

    for (int i = 0; i < num_queries; i++)
    {
        ExpRecorder qrec;
        auto qs = chrono::high_resolution_clock::now();
        partition->point_query(qrec, queries[i]);
        auto qf = chrono::high_resolution_clock::now();
        latencies.push_back(chrono::duration_cast<chrono::nanoseconds>(qf - qs).count());
        total_page_access += qrec.page_access;
    }

    // Compute metrics
    double mean_ns = accumulate(latencies.begin(), latencies.end(), 0LL) / (double)num_queries;

    vector<long long> sorted_lat = latencies;
    sort(sorted_lat.begin(), sorted_lat.end());
    double p95_ns = sorted_lat[(size_t)(0.95 * num_queries)];

    double total_query_s = accumulate(latencies.begin(), latencies.end(), 0LL) / 1e9;
    double throughput    = num_queries / total_query_s;
    double avg_page      = total_page_access / num_queries;

    delete partition;

    // ---- PRINT RESULTS ----
    cout << "\nRSMI Evaluation Results (" << distribution << " – " << cardinality << " triples)\n";
    cout << string(55, '-') << "\n";
    cout << fixed;
    cout.precision(3);
    cout << "Build Time (s):               " << build_ns / 1e9            << "\n";
    cout.precision(3);
    cout << "Index Size (MB):              " << index_bytes / (1024.0 * 1024.0) << "\n";
    cout.precision(3);
    cout << "Mean Lookup Latency (us):     " << mean_ns / 1e3             << "\n";
    cout.precision(3);
    cout << "P95 Lookup Latency (us):      " << p95_ns  / 1e3             << "\n";
    cout.precision(0);
    cout << "Query Throughput (q/s):       " << throughput                << "\n";
    cout.precision(4);
    cout << "Average Page Accesses:        " << avg_page                  << "\n";
    cout << string(55, '-') << "\n";

    return 0;
}

#endif  // use_gpu
