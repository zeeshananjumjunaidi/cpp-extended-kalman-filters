#include "kalman_filter.h"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
	MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
	x_ = x_in; // object state
	P_ = P_in; // object covariance matrix
	F_ = F_in; // state transition matrix
	H_ = H_in; // measurement matrix
	R_ = R_in; // measurement covariance matrix
	Q_ = Q_in; // process covariance matrix
}

// The Kalman filter predict function. The same for linear and extended Kalman filter
void KalmanFilter::Predict() {
	x_ = F_ * x_; // There is no external motion, so, we do not have to add "+u"
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	// Kalman filter update step. Equations from the lectures
	VectorXd y = z - H_ * x_; // error calculation
	PostUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	/**
	TODO:
	  * update the state by using Extended Kalman Filter equations
	*/
	// Recalculate x object state to rho, theta, rho_dot coordinates
	// range
	double rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
	// bearing
	double theta = atan2(x_(1), x_(0));
	// range rate
	double rho_dot = 0;

	if (fabs(rho) > 0.0001) {
		rho_dot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
	}

	VectorXd h = VectorXd(3);

	h << rho, theta, rho_dot;

	VectorXd y = z - h;
	// Calculations are essentially the same to the Update function
	PostUpdate(y);
}

// Universal update Kalman Filter step. Equations from the lectures
void KalmanFilter::PostUpdate(const VectorXd &y) {
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();

	MatrixXd PHt = P_*Ht;
	MatrixXd K = PHt *Si;

	// New state
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
