#include <filesystem>
#include <dgpsi/utilities.h>
#include <dgpsi/kernels.h>
#include <dgpsi/deep_models.h>
#include <rapidcsv.h>

using namespace dgpsi::kernels;
using namespace dgpsi::utilities;
using namespace dgpsi::deep_models::gaussian_process;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, "\t", "\n");
template <typename Derived>
void write_data(std::string name, const Eigen::MatrixBase<Derived>& matrix)
{
	std::ofstream file(name.c_str());
	file << matrix.format(CSVFormat);
}

static void write_to_file(std::string filepath, std::string line)
{
	std::ofstream myfile;
	myfile.open(filepath, std::fstream::app);
	myfile << line << "\n";
	myfile.close();
}

TMatrix read_data(std::string filename) {

	rapidcsv::Document doc(filename, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams('\t'));
	int nrows = doc.GetRowCount();
	int ncols = doc.GetColumnCount();

	TMatrix data(nrows, ncols);
	for (std::size_t i = 0; i < nrows; ++i) {
		std::vector<double> row = doc.GetRow<double>(i);
		for (std::vector<double>::size_type j = 0; j != row.size(); j++) {
			data(i, j) = row[j];
		}
	}
	return data;
}

class ProgressBar
{
	static const auto overhead = sizeof " [100%]";

	std::ostream& os;
	const std::size_t bar_width;
	std::string message;
	const std::string full_bar;

public:
	ProgressBar(std::ostream& os, std::size_t line_width,
		std::string message_, const char symbol = '|')
		: os{ os },
		bar_width{ line_width - overhead },
		message{ std::move(message_) },
		full_bar{ std::string(bar_width, symbol) + std::string(bar_width, ' ') }
	{
		if (message.size() + 1 >= bar_width || message.find('\n') != message.npos) {
			os << message << '\n';
			message.clear();
		}
		else {
			message += ' ';
		}
		write(0.0);
	}

	// not copyable
	ProgressBar(const ProgressBar&) = delete;
	ProgressBar& operator=(const ProgressBar&) = delete;

	~ProgressBar()
	{
		write(1.0);
		os << '\n';
	}

	void write(double fraction) {
		// clamp fraction to valid range [0,1]
		if (fraction < 0)
			fraction = 0;
		else if (fraction > 1)
			fraction = 1;

		auto width = bar_width - message.size();
		auto offset = bar_width - static_cast<unsigned>(width * fraction);

		os << '\r' << message;
		os.write(full_bar.data() + offset, width);
		os << " [" << std::setw(3) << static_cast<int>(100 * fraction) << "%] " << std::flush;
	}
	void write(double fraction, double nrmse) {
		// clamp fraction to valid range [0,1]
		if (fraction < 0)
			fraction = 0;
		else if (fraction > 1)
			fraction = 1;

		auto width = bar_width - message.size();
		auto offset = bar_width - static_cast<unsigned>(width * fraction);

		os << '\r' << message;
		os.write(full_bar.data() + offset, width);
		os << " [" << std::setw(3) << static_cast<int>(100 * fraction) << "%] " << " [" << std::setw(3) << std::left << std::setprecision(5) << std::fixed << nrmse * 100.0 << "%] " << std::flush;


	}
	void write(double fraction, double nrmse, double r2) {
		// clamp fraction to valid range [0,1]
		if (fraction < 0)
			fraction = 0;
		else if (fraction > 1)
			fraction = 1;

		auto width = bar_width - message.size();
		auto offset = bar_width - static_cast<unsigned>(width * fraction);

		os << '\r' << message;
		os.write(full_bar.data() + offset, width);
		os << " [" << std::setw(3) << static_cast<int>(100 * fraction) << "%] "
			<< " [NRMSE = " << std::setw(3) << std::left << std::setprecision(5) << std::fixed << nrmse * 100.0 << "%] "
			<< " [R2 = " << std::setw(3) << std::left << std::setprecision(5) << std::fixed << r2 << "]"
			<< std::flush;
	}
};

struct Case {
	Case() = default;
	Case(const std::string& problem) : problem(problem) {}
	Case(const std::string& problem, const std::string& output) : problem(problem), output(output) {}
	std::string problem;
	std::string output = "";
	unsigned int n_train;
	unsigned int experiment;
	unsigned int start;
	unsigned int finish;
	unsigned int train_iter;
	unsigned int train_impute;
	unsigned int pred_iter;
	bool plot = false;
	double likelihood_variance = 1E-10;
};

void _case1(Case& case_study, int& train_iter, int& train_impute) {
	std::string data_path = "../datasets/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/";
	auto run_problem = [=](std::string results_path, std::string exp, bool& restart) {
		TMatrix X_train = read_data(data_path + "Xsc_train.dat");
		TMatrix Y_train = read_data(data_path + "Y_train.dat");

		//TMatrix X_test = read_data(data_path + "Xsc_test.dat");
		//TMatrix Y_test = read_data(data_path + "Y_test.dat");

		TMatrix X_plot = read_data(data_path + "X_plot.dat");
		TMatrix X_test = read_data(data_path + "X_mcs.dat");
		TMatrix Y_test = read_data(data_path + "Y_mcs.dat");

		Graph graph(std::make_pair(X_train, Y_train), 1);
		for (unsigned int i = 0; i < graph.n_layers; ++i) {
			TVector ls = TVector::Constant(X_train.cols(), 1.0);
			graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52, ls);
			graph.layer(static_cast<int>(i))->set_likelihood_variance(case_study.likelihood_variance);
			graph.layer(static_cast<int>(i))->fix_likelihood_variance();
		}
		DGPSI model(graph);
		model.train(train_iter, train_impute);
		bool nanflag = false;

		std::string m_path = results_path + exp + "-M.dat";
		std::string v_path = results_path + exp + "-V.dat";
		std::string mcs_path = results_path + exp;
		MatrixPair Z = model.predict(X_test, Y_test, mcs_path, nanflag, case_study.pred_iter, 96);
		TMatrix mean = Z.first;
		TMatrix var = Z.second;
		double nrmse = metrics::rmse(Y_test, mean, true);

		if (nanflag) {
			restart = true;
		}
		else {
			std::string e_path = results_path + "NRMSE.dat";
			std::cout << "NRMSE = " << nrmse << std::endl;
			write_data(m_path, mean);
			write_data(v_path, var);
			write_to_file(e_path, std::to_string(nrmse));

			// Plot
			MatrixPair Zplot = model.predict(X_plot, case_study.pred_iter, 96);
			TMatrix mplt = Zplot.first;
			TMatrix vplt = Zplot.second;
			std::string mplt_path = results_path + exp + "-M-PLT.dat";
			std::string vplt_path = results_path + exp + "-V-PLT.dat";
			write_data(mplt_path, mplt);
			write_data(vplt_path, vplt);


		}
	};


	if (!std::filesystem::exists("../results/case_1/"))
		std::filesystem::create_directory("../results/case_1/");
	// ../results/case_1/analytic2
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem);
	// ../results/case_1/analytic2/25
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train)))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train));
	// ../results/case_1/analytic2/25/100
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter)))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter));

	// ../results/case_1/analytic2/25/100/100 || TRAIN_ITER/TRAIN_IMPUTE
	std::string main_results_path = "../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter) + "/" + std::to_string(train_impute);
	if (!std::filesystem::exists(main_results_path)) std::filesystem::create_directory(main_results_path);
	unsigned int ii = case_study.start;
	while (true) {
		std::cout << "================= " << "Running " << case_study.problem << "-" << case_study.n_train << " :"
			<< train_iter << "-" << train_impute << "================= " << std::endl;
		bool restart = false;
		std::cout << "================= " << "" << " REP " << ii << " ================" << std::endl;
		// ../results/case_1/analytic2/25/100/100/1 .... 25 || TRAIN_ITER/TRAIN_IMPUTE/REP
		std::string results_path = main_results_path + "/" + std::to_string(ii) + "/";
		if (!std::filesystem::exists(results_path)) std::filesystem::create_directory(results_path);
		run_problem(results_path, std::to_string(case_study.experiment), restart);
		if (restart) {
			std::system("clear");
			continue;
		}
		else ii++;
		if (ii == case_study.finish) break;

	}
}

void case1() {

	std::vector<int> train_iter = { 500 };
	std::vector<int> train_impute = { 900 };
	
	Case AN_C1_1("analytic2");
	AN_C1_1.n_train = 25;
	AN_C1_1.experiment = 3;
	AN_C1_1.start = 1;
	AN_C1_1.finish = 2;
	AN_C1_1.pred_iter = 500;
	AN_C1_1.likelihood_variance = 1E-3;

	for (int ii : train_iter) {
		for (int jj : train_impute) {
			_case1(AN_C1_1, ii, jj);
		}
	}
}

int main() {
	case1();
	return 0;
}