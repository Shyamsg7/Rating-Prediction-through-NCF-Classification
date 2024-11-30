Rating prediction using Neural collabarative filtering 

Neural collaborative filtering(NCF), is a deep learning based framework for making recommendations. The key idea is to learn the user-item interaction using neural networks. Check the follwing paper for details about NCF.

He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
This code is highly inspired by this GitHub repository.(Whole credit goes to that person)

This model is sligthly altered to perform rating prediction through classification setting by addding a classification layer at the end of original architecture.

Usage
cd src ( Move to the src folder which consists of the whole code)

python3 train.py (download the necessary requirements if there is any thing to be installed)
