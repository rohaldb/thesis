#include "data.h"

string Data::data_folder = "";
string Data::dataset = "audio";
string Data::index_folder = "index/";
string Data::result_folder = "result/";

string Data::data_file = "";
string Data::query_file = "";
string Data::index_file = "";
string Data::result_file = "";

void Data::read_arguments(int argc, char* argv[])
{
    for(int i = 1; i + 1 < argc; i += 2)
    {
        string argName = string(argv[i]);
        istringstream argstream(argv[i + 1]);

        if(argName == "data")
            argstream >> data_folder;
        else if(argName == "dataset")
            argstream >> dataset;
        else if(argName == "index")
            argstream >> index_folder;
        else if(argName == "result")
            argstream >> result_folder;
        else if(argName == "t")
            argstream >> t;
        else if(argName == "k")
            argstream >> k;
        else if(argName == "hull")
            argstream >> hull;
        else if(argName == "p")
            argstream >> p;
        else if(argName == "lim")
            argstream >> lim;
        else if(argName == "r")
            argstream >> r;
        else
            assert(false);
    }

    data_file = data_folder + dataset + "_base.fvecs";
    query_file = data_folder + dataset + "_query.fvecs";
    index_file = index_folder + dataset + "_" + to_string(t);
    result_file = result_folder + dataset
                  + "_t" + to_string(t) + "_k" + to_string(k) 
                  + "_hull" + to_string(hull) + "_p" + to_string(p)
                  + "_lim" + to_string(lim) + "_r" + to_string(r);
}

void Data::print_arguments()
{
    cout << "data folder : " << data_folder << endl;
    cout << "dataset : " << dataset << endl;
    cout << "index folder : " << index_folder << endl;
    cout << "result folder : " << result_folder << endl;
    cout << "data file : " << data_file << endl;
    cout << "query file : " << query_file << endl;
    cout << "index file : " << index_file << endl;
    cout << "result file : " << result_file << endl;
    cout << "t : " << t << endl;
    cout << "k : " << k << endl;
    cout << "hull : " << hull << endl;
    cout << "p : " << p << endl;
    cout << "lim : " << lim << endl;
    cout << "r : " << r << endl;
}

vector<float_v> Data::data;
vector<float_v> Data::query;
int Data::d = 192;
int Data::n = 53387;
int Data::m = 200;

void Data::read_data()
{
    FILE* f = fopen(data_file.c_str(), "rb");
    assert(f != NULL);
    size_t a = fread(&d, sizeof(int), 1, f);
    fseek(f, 0, SEEK_END);
    off_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    n = (int)(size / (sizeof(int)+ sizeof(float)*d));
    float* point = new float[d];
    for(int i = 0; i < n; i++)
    {
        a = fread(&d, sizeof(int), 1, f);
        a = fread(point, sizeof(float), d, f);
        vector<float> vec(point, point + d);
        data.push_back(vec);
    }
    fclose(f);
    assert(data.size() == n);
    for(int i = 0; i < n; i++)
        assert(data[i].size() == d);
    Data::LOG << "read data " << n << " * " << d << endl;
    Data::LOG << "data[0][:10]";
    cout << "read data " << n << " * " << d << endl;
    cout << "data[0][:10]";
    for(int i = 0; i < d && i < 10; i++)
    {
        Data::LOG << " " << data[0][i];
        cout << " " << data[0][i];
    }
    Data::LOG << endl;
    cout << endl; 
}

void Data::read_query()
{
    FILE* f = fopen(query_file.c_str(), "rb");
    assert(f != NULL);
    size_t a = fread(&d, sizeof(int), 1, f);
    fseek(f, 0, SEEK_END);
    off_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    m = (int)(size / (sizeof(int)+ sizeof(float)*d));
    float* point = new float[d];
    for(int i = 0; i < m; i++)
    {
        a = fread(&d, sizeof(int), 1, f);
        a = fread(point, sizeof(float), d, f);
        vector<float> vec(point, point + d);
        query.push_back(vec);
    }
    fclose(f);
    assert(query.size() == m);
    for(int i = 0; i < m; i++)
        assert(query[i].size() == d);
    Data::LOG << "read query " << m << " * " << d << endl;
    Data::LOG << "query[0][:10]";
    cout << "read query " << m << " * " << d << endl;
    cout << "query[0][:10]";
    for(int i = 0; i < d && i < 10; i++)
    {
        Data::LOG << " " << query[0][i];
        cout << " " << query[0][i];
    }
    Data::LOG << endl;
    cout << endl;
}

int Data::t = 4;

int Data::k = 20;
int Data::hull = 100;
int Data::p = 2;
int Data::lim = 2000;
int Data::r = 10;

ofstream Data::LOG;

void Data::open_output_file()
{
    LOG.open(result_file.c_str());
    assert(LOG.is_open());
    LOG << "dataset : " << dataset << endl;
    LOG << "t : " << t << endl;
    LOG << "k : " << k << endl;
    LOG << "hull : " << hull << endl;
    LOG << "p : " << p << endl;
    LOG << "lim : " << lim << endl;
    LOG << "r : " << r << endl;
}
void Data::close_output_file()
{
    LOG.close();
}

float dot_product(vector<float>& p1, vector<float>& p2)
{
    float sum = 0;
    for(int i = 0; i < Data::d; i++)
        sum += p1[i] * p2[i];
    return sum;
}

float dist2(vector<float>& p1, vector<float>& p2)
{
    float sum = 0;
    for(int i = 0; i < Data::d; i++)
        sum += (p1[i]-p2[i]) * (p1[i]-p2[i]);
    return sum;
}

float dist(vector<float>& p1, vector<float>& p2)
{
    return sqrt(dist2(p1, p2));
}

void BF::knn(vector<float>& q, vector<int>& rst)
{
    min_heap mh;
    for(int i = 0; i < Data::n; i++)
    {
        float d2 = dist2(Data::data[i], q);
        if(d2 == 0)
            continue;
        mh.emplace(d2, i);
    }
    for(int i = 0; i < Data::k && !mh.empty(); i++)
    {
        auto iter = mh.top();
        const MinHeapNode& cur = iter;
        mh.pop();
        rst.push_back(cur.id);
    }
}
