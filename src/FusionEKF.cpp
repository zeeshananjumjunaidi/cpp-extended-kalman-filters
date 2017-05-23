#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Noise covariance matrix computation, value of 9.0 is given
float noise_ax = 9.0f;
float noise_ay = 9.0f;



FusionEKF::FusionEKF() {
	is_initialized_ = false;
	//previous_timestamp_ = 0;
	ekf_.x_ = VectorXd(4);
	ekf_.x_ << 1, 1, 1, 1;
	// Initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	H_laser_ << 1, 0, 0, 0,
		0, 1, 0, 0;
	Hj_ = MatrixXd(3, 4);
	Hj_ << 1, 1, 0, 0,
		1, 1, 0, 0,
		1, 1, 1, 1;
	// There is no need to tune R for this project because it is given in the task.
	// Measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
		0, 0.0225;

	// Measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
		0, 0.0009, 0,
		0, 0, 0.09;
	/**
	TODO:
	  * Finish initializing the FusionEKF.
	  * Set the process and measurement noises
	*/
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		/**
		TODO:
		  * Initialize the state ekf_.x_ with the first measurement.
		  * Create the covariance matrix.
		  * Remember: you'll need to convert radar from polar to cartesian coordinates.
		*/
		ekf_.x_ = VectorXd(4);
		ekf_.x_ << 1, 1, 1, 1;

		// initial transition matrix F_
		ekf_.F_ = MatrixXd::Identity(4, 4);

		// first measurement
		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			/**
			Convert radar from polar to cartesian coordinates and initialize state.
			*/
			float rho = measurement_pack.raw_measurements_[0]; // range
			float phi = measurement_pack.raw_measurements_[1]; // bearing
			float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho

			// Coordinates convertion from polar to cartesian
			float x = rho		* cos(phi);
			float y = rho		* sin(phi);
			float vx = rho_dot	* cos(phi);
			float vy = rho_dot	* sin(phi);
			ekf_.x_ << x, y, vx, vy;
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			// We don't know the velociities in first measurement
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
		}

		// Initial covariance matrix
		ekf_.P_ = MatrixXd(4, 4);
		ekf_.P_ << 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1000, 0,
			0, 0, 0, 1000;

		// Laser measurement matrix
		H_laser_ << 1, 0, 0, 0,
			0, 1, 0, 0;

		// Radar measurement matrix
		Hj_ << 1, 1, 0, 0,
			1, 1, 0, 0,
			1, 1, 1, 1;

		// Save the initial timestamp for dt calculation
		previous_timestamp_ = measurement_pack.timestamp_;

		// Done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/

	 // Calculate the timestep between measurements in seconds
	double dt = (measurement_pack.timestamp_ - previous_timestamp_);
	dt /= 1000000.0; // convert microseconds to s
	previous_timestamp_ = measurement_pack.timestamp_;

	// State transition matrix update
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;


	// Precompute repeated values of Q
	float dt_2 = dt   * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	//Process noise covariance matrix Q
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << dt_4 / 4.0*noise_ax, 0.0, dt_3 / 2.0*noise_ax, 0.0,
		0.0, dt_4 / 4.0*noise_ay, 0.0, dt_3 / 2.0*noise_ay,
		dt_3 / 2.0*noise_ax, 0.0, dt_2*noise_ax, 0.0,
		0.0, dt_3 / 2.0*noise_ay, 0.0, dt_2*noise_ay;

	ekf_.Predict();

	/****************************************************************************
	***  Update ***
	****************************************************************************/

	// Check input sensor type and update accordingly
	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		// Use Jacobian instead of H
		Hj_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.H_ = Hj_;
		ekf_.R_ = R_radar_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	}
	else {
		// Laser updates
		ekf_.H_ = H_laser_;
		ekf_.R_ = R_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
	}
	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}
