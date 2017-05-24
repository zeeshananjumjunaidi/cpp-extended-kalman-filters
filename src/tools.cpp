#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

// Calculate root mean squared error
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
	const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;
	// Check the validity of the following inputs:
	// The estimation vector size should not be zero
	if (estimations.size() == 0) {
		cout << "Input is empty" << endl;
		return rmse;
	}
	// The estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size()) {
		cout << "Invalid estimation or ground_truth. Data should have the same size" << endl;
		return rmse;
	}
	// Accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {
		VectorXd residual = estimations[i] - ground_truth[i];
		// Coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	// Calculate the mean
	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();
	return rmse;
}

// Calculate Jacobian Matrix
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	double c1 = (px * px) + (py * py);
	double c2 = sqrt(c1);
	double c3 = (c1 * c2);

	//check division by zero
	if (fabs(c1) < 0.0000001) {
		Hj << 0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0;
	}
	else {
		float pxc2 = px / c2;
		float pyc2 = py / c2;
		float pyvx = vy * px;
		float pxvy = vx * py;
		//compute the Jacobian matrix
		Hj << pxc2, pyc2, 0, 0,
			-(py / c1), (px / c1), 0, 0,
			py * (pxvy - pyvx) / c3, px * (pxvy - pyvx) / c3, pxc2, pyc2;
	}
	return Hj;
}