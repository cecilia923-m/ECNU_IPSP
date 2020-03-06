# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '仿真平台UI设计.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!
# 主窗口设计

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import os
from math import *
import numpy as np
from scipy import interpolate
from numpy.linalg import *
import pandas as pd
import csv
import warnings
import matplotlib.pyplot as plt

# from visualization import visualization
import seaborn as sns


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("室内定位仿真平台")
        Form.resize(1687, 1294)


        self.input_Button = QtWidgets.QPushButton(Form)
        self.input_Button.setGeometry(QtCore.QRect(430, 70, 150, 46))
        self.input_Button.setObjectName("input_Button")
        self.input_Button.setToolTip("请输入csv文件")

        # 点击输入文件按钮时，链接到文件槽，并在TextEdit中显示文件名
        self.input_Button.clicked.connect(self.read_file)

        self.File_name = QtWidgets.QTextEdit(Form)
        self.File_name.setGeometry(QtCore.QRect(630, 70, 681, 40))
        self.File_name.setObjectName("File_name")
        self.File_name.setPlaceholderText("请传入csv格式的Tdoa文件")

        self.Algorithm_label = QtWidgets.QLabel(Form)
        self.Algorithm_label.setGeometry(QtCore.QRect(460, 160, 108, 24))
        self.Algorithm_label.setObjectName("Algorithm_label")

        self.Algorithm_comboBox = QtWidgets.QComboBox(Form)
        self.Algorithm_comboBox.setGeometry(QtCore.QRect(630, 149, 251, 41))
        self.Algorithm_comboBox.setObjectName("Algorithm_comboBox")
        self.Algorithm_comboBox.addItem("请选择")
        self.Algorithm_comboBox.addItem("Chan")
        # self.Algorithm_comboBox.addItem("Fang")
        self.Algorithm_comboBox.addItem("Chan & Taylor")
        # self.Algorithm_comboBox.addItem("Chan and Taylor")
        # self.Algorithm_comboBox.currentIndexChanged.connect(self.selectionChange) # 当前索引变化所绑定的槽

        self.realpos_label = QtWidgets.QLabel(Form)
        self.realpos_label.setGeometry(QtCore.QRect(460, 230, 108, 24))
        self.realpos_label.setObjectName("real_pos_label")

        self.X_label = QtWidgets.QLabel(Form)
        self.X_label.setGeometry(QtCore.QRect(630, 230, 108, 24))
        self.X_label.setObjectName("X_label")

        # 设置一个浮点数校验器
        self.doubleValidator = QDoubleValidator()

        self.X_lineEdit = QtWidgets.QLineEdit(Form)
        self.X_lineEdit.setGeometry(QtCore.QRect(670, 220, 181, 41))
        self.X_lineEdit.setObjectName("X_lineEdit")
        self.X_lineEdit.setPlaceholderText("浮点型")
        self.X_lineEdit.setValidator(self.doubleValidator)

        self.Y_label = QtWidgets.QLabel(Form)
        self.Y_label.setGeometry(QtCore.QRect(930, 230, 108, 24))
        self.Y_label.setObjectName("Y_label")

        self.Y_lineEdit = QtWidgets.QLineEdit(Form)
        self.Y_lineEdit.setGeometry(QtCore.QRect(970, 220, 161, 41))
        self.Y_lineEdit.setObjectName("Y_lineEdit")
        self.Y_lineEdit.setPlaceholderText("浮点型")
        self.Y_lineEdit.setValidator(self.doubleValidator)

        # 当输入坐标每发生一次变化时，触发get_x_y函数，返回x，y值
        # self.realpos_lineEdit.editingFinished(self.get_x_y)

        self.cal_error_btn = QtWidgets.QPushButton(Form)
        self.cal_error_btn.setGeometry(QtCore.QRect(1010, 310, 201, 51))
        self.cal_error_btn.setObjectName("cal_error_btn")
        self.cal_error_btn.clicked.connect(self.cal_error)

        self.cal_pos_btn = QtWidgets.QPushButton(Form)
        self.cal_pos_btn.setGeometry(QtCore.QRect(390, 305, 191, 51))
        self.cal_pos_btn.setObjectName("cal_pos_btn")
        self.cal_pos_btn.clicked.connect(self.cal_position)

        self.position_textEdit = QtWidgets.QTextEdit(Form)
        self.position_textEdit.setGeometry(QtCore.QRect(190, 400, 641, 681))
        self.position_textEdit.setObjectName("position_textEdit")

        self.error_textEdit = QtWidgets.QTextEdit(Form)
        self.error_textEdit.setGeometry(QtCore.QRect(1000, 400, 411, 681))
        self.error_textEdit.setObjectName("error_textEdit")

        self.visualization_btn = QtWidgets.QPushButton(Form)
        self.visualization_btn.setGeometry(QtCore.QRect(1320, 1110, 251, 91))
        self.visualization_btn.clicked.connect(self.visualization)

        self.CDF_Image_btn = QtWidgets.QPushButton(Form)
        self.CDF_Image_btn.setGeometry(QtCore.QRect(1230, 310, 171, 51))
        self.CDF_Image_btn.setObjectName("CDF_Button")
        self.CDF_Image_btn.clicked.connect(self.CDF_Curve)

        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.visualization_btn.setFont(font)
        self.visualization_btn.setObjectName("visualization_btn")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "室内定位仿真平台"))
        Form.setWindowIcon(QIcon('./ECNU.png'))
        self.input_Button.setText(_translate("Form", "输入文件"))
        self.Algorithm_label.setText(_translate("Form", "算法选择："))
        self.realpos_label.setText(_translate("Form", "真实坐标："))
        self.cal_error_btn.setText(_translate("Form", "坐标误差解算"))
        self.cal_pos_btn.setText(_translate("Form", "坐标解算"))
        self.visualization_btn.setText(_translate("Form", "结果可视化"))
        self.X_label.setText(_translate("Form", "X:"))
        self.Y_label.setText(_translate("Form", "Y:"))
        self.CDF_Image_btn.setText(_translate("Form","生成CDF"))

    def tdoa_process(self,file_name):
        data_SYNC = []
        data_TAG = []

        anchor = ['416', '448', '455', '503']

        with open(file_name) as f:
            data = csv.reader(f)
            for line in data:
                if line[1] == '$SYNC':
                    if line[4] in anchor:  # line4指的是从基站
                        data_SYNC.append(line)
                else:
                    if line[3] in anchor:
                        data_TAG.append(line)

        data_SYNC = pd.DataFrame(data_SYNC)
        data_SYNC = data_SYNC.iloc[:, [0, 3, 4, 5, 6, 7, 9, 10]]
        data_SYNC.columns = ['dateTime', 'series_num', 'base_ID', 'master_ID', 'master_tick', 'sub_tick', 'fp',
                             'rx']
        data_SYNC['dateTime'] = pd.to_datetime(data_SYNC['dateTime'], format='-%Y-%m-%d %H:%M:%S.%f')
        data_SYNC = data_SYNC.groupby('dateTime').apply(
            lambda x: x.sort_values('base_ID'))  # 按照datatime的值聚集，相同datatime的列再按照base_ID排序
        data_SYNC.set_index('dateTime', inplace=True)  # 列索引设置成datetime，直接对原始对象修改
        data_SYNC = data_SYNC.astype('float')

        data_TAG = pd.DataFrame(data_TAG)
        data_TAG = data_TAG.iloc[:, [0, 3, 4, 5, 6, 12, 13, 14]]
        data_TAG.columns = ['dateTime', 'base_ID', 'tag_ID', 'base_tick', 'series_num', 'package_num', 'fp',
                            'rx']
        data_TAG['dateTime'] = pd.to_datetime(data_TAG['dateTime'], format='-%Y-%m-%d %H:%M:%S.%f')
        data_TAG = data_TAG.groupby('dateTime').apply(lambda x: x.sort_values('base_ID'))
        data_TAG.set_index('dateTime', inplace=True)
        data_TAG = data_TAG.astype('float')

        time_SYNC = data_SYNC.index.unique()  # 取到目录数组特征值 datatime
        time_TAG = data_TAG.index.unique()

        for t in time_SYNC:  # 某时刻中不是4个基站都有数据的扔掉（不能拿来计算）
            if data_SYNC.loc[t].shape[0] != 4:
                data_SYNC.drop(t, inplace=True)
        for t in time_TAG:
            if data_TAG.loc[t].shape[0] != 4:
                data_TAG.drop(t, inplace=True)
        time_SYNC = data_SYNC.index.unique()
        time_TAG = data_TAG.index.unique()

        tdoa = []
        for t_sync1, t_sync2, t_sync3 in zip(time_SYNC[::3], time_SYNC[1::3], time_SYNC[2::3]):  # 3组一取，用来算MLT
            t_tag_bool = (time_TAG > t_sync1) & (time_TAG < t_sync3)
            t_tag = time_TAG[t_tag_bool]

            if t_tag.shape[0] != 0:  # 把两次同步中的定位包的第一个拿过来取做SLT
                t_tag = t_tag[0]
                ST1 = data_SYNC.loc[t_sync1, 'sub_tick'].values
                ST2 = data_SYNC.loc[t_sync2, 'sub_tick'].values
                MT1 = data_SYNC.loc[t_sync2, 'master_tick'].values
                MT2 = data_SYNC.loc[t_sync3, 'master_tick'].values
                SLT = data_TAG.loc[t_tag, 'base_tick'].values

                MLT = (MT2 - MT1) * (SLT - ST1) / (ST2 - ST1) + MT1
                MLT[2] = SLT[2]  # 0-3中 2为主基站
                Distance_TDOA = (MLT - MLT[2]) * 0.469
                tdoa.append(Distance_TDOA)

        tdoa = np.array(tdoa)
        del_num = []
        for i in range(len(tdoa)):
            if (np.abs(tdoa[i]) > 500).any():
                del_num.append(i)
        tdoa = np.delete(tdoa, del_num, 0)
        return tdoa

    def Pos_cal_xy_TDOA(self,AN, D_Aij_T):
        A = []
        B = []
        C = []
        pos_xy = []

        for i in range(len(D_Aij_T)):  # 矩阵的设置等详见锚节点差分推导
            A.append(((AN[i + 1][0] - AN[0][0]), (AN[i + 1][1] - AN[0][1]), D_Aij_T[i]))
            B.append(
                (AN[i + 1][0] ** 2 + AN[i + 1][1] ** 2 - AN[0][0] ** 2 - AN[0][1] ** 2 - D_Aij_T[i] ** 2) / 2)

        A = np.array(A)
        B = np.array(B)
        C = inv((A.T).dot(A)).dot((A.T)).dot(B)

        pos_xy.append(C[0])
        pos_xy.append(C[1])

        return pos_xy

    def Taylor_xy_three_TDOA(self,Pos_xy, D_Aij_T, AN):
        bre = 0
        D_xy = (0, 0)  # 坐标微小量
        D_xy = np.array(D_xy)

        tag1_xy = []
        tag1_xy.append(Pos_xy[0])
        tag1_xy.append(Pos_xy[1])
        # tag1_xy.append(Pos_xy[2])
        # tag1_xy.append(0.856)
        tag1_xy = np.array(tag1_xy)  # tag的xy轴坐标

        R21 = D_Aij_T[0]  # 405-475 TDOA
        R31 = D_Aij_T[1]  # 446-475 TDOA
        R41 = D_Aij_T[2]  # 486-475 TDOA

        err_xy_taylor = 1  # abs(D[0])+abs(D[1]) 决定迭代次数

        Ht_xy = []
        R1_xy = sqrt(
            (AN[0][0] - tag1_xy[0]) ** 2 + (AN[0][1] - tag1_xy[1]) ** 2)  # AN1与tag的距离
        R2_xy = sqrt(
            (AN[1][0] - tag1_xy[0]) ** 2 + (AN[1][1] - tag1_xy[1]) ** 2)  # AN2与tag的距离
        R3_xy = sqrt(
            (AN[2][0] - tag1_xy[0]) ** 2 + (AN[2][1] - tag1_xy[1]) ** 2)  # AN3与tag的距离
        R4_xy = sqrt(
            (AN[3][0] - tag1_xy[0]) ** 2 + (AN[3][1] - tag1_xy[1]) ** 2)  # AN4与tag的距离

        Ht_xy.append(R21 - (R2_xy - R1_xy))
        Ht_xy.append(R31 - (R3_xy - R1_xy))
        Ht_xy.append(R41 - (R4_xy - R1_xy))

        Ht_xy = np.array(Ht_xy)  # 观测TDOA（扣除误差）-几何TDOA

        Gt_xy = []
        Gt0_xy = (((AN[0][0] - tag1_xy[0]) / R1_xy) - ((AN[1][0] - tag1_xy[0]) / R2_xy),
                  ((AN[0][1] - tag1_xy[1]) / R1_xy) - ((AN[1][1] - tag1_xy[1]) / R2_xy))
        Gt1_xy = (((AN[0][0] - tag1_xy[0]) / R1_xy) - ((AN[2][0] - tag1_xy[0]) / R3_xy),
                  ((AN[0][1] - tag1_xy[1]) / R1_xy) - ((AN[2][1] - tag1_xy[1]) / R3_xy))
        Gt2_xy = (((AN[0][0] - tag1_xy[0]) / R1_xy) - ((AN[3][0] - tag1_xy[0]) / R4_xy),
                  ((AN[0][1] - tag1_xy[1]) / R1_xy) - ((AN[3][1] - tag1_xy[1]) / R4_xy))

        Gt_xy.append(Gt0_xy)
        Gt_xy.append(Gt1_xy)
        Gt_xy.append(Gt2_xy)

        Gt_xy = np.array(Gt_xy)

        num_xy_taylor = 0
        while err_xy_taylor > 0.000001:  # 0.000001
            D_xy = inv((Gt_xy.T).dot(Gt_xy)).dot((Gt_xy.T)).dot(Ht_xy)

            tag1_xy = tag1_xy + D_xy

            err_xy_taylor = abs(D_xy[0]) + abs(D_xy[1])
            R1_xy = sqrt(
                (AN[0][0] - tag1_xy[0]) ** 2 + (AN[0][1] - tag1_xy[1]) ** 2)  # AN1与tag的距离
            R2_xy = sqrt(
                (AN[1][0] - tag1_xy[0]) ** 2 + (AN[1][1] - tag1_xy[1]) ** 2)  # AN2与tag的距离
            R3_xy = sqrt(
                (AN[2][0] - tag1_xy[0]) ** 2 + (AN[2][1] - tag1_xy[1]) ** 2)  # AN3与tag的距离
            R4_xy = sqrt(
                (AN[3][0] - tag1_xy[0]) ** 2 + (AN[3][1] - tag1_xy[1]) ** 2)  # AN4与tag的距离

            Ht_xy[0] = R21 - (R2_xy - R1_xy)
            Ht_xy[1] = R31 - (R3_xy - R1_xy)
            Ht_xy[2] = R41 - (R4_xy - R1_xy)

            Gt0_xy = (((AN[0][0] - tag1_xy[0]) / R1_xy) - ((AN[1][0] - tag1_xy[0]) / R2_xy),
                      ((AN[0][1] - tag1_xy[1]) / R1_xy) - ((AN[1][1] - tag1_xy[1]) / R2_xy))
            Gt1_xy = (((AN[0][0] - tag1_xy[0]) / R1_xy) - ((AN[2][0] - tag1_xy[0]) / R3_xy),
                      ((AN[0][1] - tag1_xy[1]) / R1_xy) - ((AN[2][1] - tag1_xy[1]) / R3_xy))
            Gt2_xy = (((AN[0][0] - tag1_xy[0]) / R1_xy) - ((AN[3][0] - tag1_xy[0]) / R4_xy),
                      ((AN[0][1] - tag1_xy[1]) / R1_xy) - ((AN[3][1] - tag1_xy[1]) / R4_xy))

            Gt_xy[0] = Gt0_xy
            Gt_xy[1] = Gt1_xy
            Gt_xy[2] = Gt2_xy

            num_xy_taylor += 1
            if num_xy_taylor > 200000:
                bre = 1
                print("break_taylor")
                print(num_xy_taylor)
                break

        xy = ((tag1_xy[0], tag1_xy[1]))
        xy = np.array(xy)

        return xy, bre

    # def Err_sqrt(XYZ_estimate):  # 计算误差
    #     Err_sqrt_xy = []
    #     Err_sqrt_xy.append(sqrt((XYZ_estimate[0] - float(self.X_lineEdit.text()))) ** 2 + (XYZ_estimate[1] - XYZ_real[1]) ** 2))
    #     return Err_sqrt_xy

    # 读取文件的槽
    def read_file(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)  # 设置文件模式，打开任意类型的文件
        dialog.setFilter(QDir.Files)  # 过滤，选则文件，打开选中的文件


        if dialog.exec():  # 打开显示对话框
            all_filename = dialog.selectedFiles()
            # 读取文件名字，并显示在textEdit中
            (path,filename) = os.path.split(str(all_filename))
            self.File_name.setText(filename[:-2])
            # self.File_name.setText(str(all_filename).split('/')[-1]) #分割文件名的另一种方法，使用split
            # print(type(all_filename))
            # self.File_name.adjustSize()

        # self.Tdoa = self.tdoa_process() #'g1-8.23.csv'

    # def selectionChange(self):
    #     # self.Tdoa = self.tdoa_process(self.File_name.toPlainText())  # 'g1-8.23.csv'
    #     pass

    # 设置计算坐标的槽


    def Err_sqrt(self,XYZ_estimate):  # 计算误差

        Err_sqrt_xy = []
        Err_sqrt_xy.append(sqrt((XYZ_estimate[0] - float(self.X_lineEdit.text())) ** 2 + (XYZ_estimate[1] - float(self.Y_lineEdit.text())) ** 2))
        return Err_sqrt_xy

    def cal_position(self):
        self.Tdoa = self.tdoa_process(self.File_name.toPlainText())
        self.AN = []
        self.taylor_plot = []
        self.chan_plot = []
        self.Taylor_error_plot = []
        self.Chan_error_plot = []

        # 407房间的基站坐标
        self.AN.append((21.905, 7.731))  # 455坐标
        self.AN.append((22.302, 3.555))  # 416坐标
        self.AN.append((15.240, 3.555))  # 448坐标
        self.AN.append((15.382, 7.731))  # 503坐标

        self.data_num = 300

        if self.Algorithm_comboBox.currentText() == "Chan":
            for i in range(0, self.data_num, 1):
                Tdoa_tem = []
                Tdoa_dis21 = Tdoa_dis31 = Tdoa_dis41 = 0

                for j in range(i, 10 + i, 1):  # 20组TDOA数据为一大组，整合
                    Tdoa_dis21 += self.Tdoa[j][0] / 1000  # 416-455
                    Tdoa_dis31 += self.Tdoa[j][1] / 1000  # 448-455
                    Tdoa_dis41 += self.Tdoa[j][3] / 1000  # 503-455
                Tdoa_tem.append(Tdoa_dis21)
                Tdoa_tem.append(Tdoa_dis31)
                Tdoa_tem.append(Tdoa_dis41)
                tdoa = Tdoa_tem

                self.Chan_xy = self.Pos_cal_xy_TDOA(self.AN, tdoa)
                self.chan_plot.append(self.Chan_xy)
                self.Chan_Error = self.Err_sqrt(self.Chan_xy)
                self.Chan_error_plot.append(self.Chan_Error)
                # chan_array = np.array(chan_plot)
                # print('第', i + 1, '次结算结果：', self.Chan_xy)
            self.position_textEdit.setText(str(self.chan_plot))

        elif self.Algorithm_comboBox.currentText() == "Chan & Taylor":
            for i in range(0, self.data_num, 1):
                Tdoa_tem = []
                Tdoa_dis21 = Tdoa_dis31 = Tdoa_dis41 = 0

                for j in range(i, 10 + i, 1):  # 20组TDOA数据为一大组，整合
                    Tdoa_dis21 += self.Tdoa[j][0] / 1000  # 416-455
                    Tdoa_dis31 += self.Tdoa[j][1] / 1000  # 448-455
                    Tdoa_dis41 += self.Tdoa[j][3] / 1000  # 503-455
                Tdoa_tem.append(Tdoa_dis21)
                Tdoa_tem.append(Tdoa_dis31)
                Tdoa_tem.append(Tdoa_dis41)
                tdoa = Tdoa_tem

                Taylor_initial_Pos = [(self.AN[0][0] + self.AN[1][0] + self.AN[2][0] + self.AN[3][0]) / 4,
                                      (self.AN[0][1] + self.AN[1][1] + self.AN[2][1] + self.AN[3][1]) / 4]
                self.Taylor_xy, bre = self.Taylor_xy_three_TDOA(Taylor_initial_Pos, tdoa, self.AN)
                self.taylor_plot.append(self.Taylor_xy)
                # self.taylor_array = np.array(self.taylor_plot)
                # print('第', i + 1, '次结算结果：', self.Taylor_xy)
                self.Taylor_Error = self.Err_sqrt(self.Taylor_xy)
                self.Taylor_error_plot.append(self.Taylor_Error)
            self.position_textEdit.setText(str(self.taylor_plot))

        pass

    # 设置计算误差的槽

    def cal_error(self):
        if self.Algorithm_comboBox.currentText() == "Chan":

            self.error_textEdit.setText(str(self.Chan_error_plot))

        elif self.Algorithm_comboBox.currentText() == "Chan & Taylor":

            self.error_textEdit.setText(str(self.Taylor_error_plot))


    # 可视化按钮连接的槽
    def visualization(self):
        AN_array = np.array(self.AN)  # 基站的坐标
        if self.Algorithm_comboBox.currentText() == "Chan":
            chan_array = np.array(self.chan_plot)
            all_array = np.vstack((chan_array, AN_array))  # 将基站坐标和计算的位置坐标组合在一起
            plt.scatter(AN_array[:, 0], AN_array[:, 1], marker='o', color='red', s=90, label='BaseStation_Position')
            plt.scatter(chan_array[:, 0], chan_array[:, 1], marker='+', color='orange', s=40, label='Chan_Cal_Position')
            plt.scatter(float(self.X_lineEdit.text()), float(self.Y_lineEdit.text()), marker='x', color='blue', s=50,
                        label='real_position')
            plt.title("Chan_Algorithm",fontsize = 22)
            plt.legend(loc='best')
            plt.show()

        elif self.Algorithm_comboBox.currentText() == "Chan & Taylor":
            taylor_array = np.array(self.taylor_plot)
            all_array = np.vstack((taylor_array, AN_array))  # 将基站坐标和计算的位置坐标组合在一起
            plt.scatter(AN_array[:, 0], AN_array[:, 1], marker='o', color='red', s=90, label='BaseStation_Position')
            plt.scatter(taylor_array[:, 0], taylor_array[:, 1], marker='+', color='orange', s=40, label='Taylor_Cal_Position')
            plt.scatter(float(self.X_lineEdit.text()),float(self.Y_lineEdit.text()),marker='x', color='blue', s=50,
                        label='real_position')
            plt.title("Chan & Taylor_Algorithm",fontsize = 22)
            plt.legend(loc='best')
            plt.show()


    # 生成CDF误差曲线
    def CDF_Curve(self):
        if self.Algorithm_comboBox.currentText() == "Chan":

            Chan_error_array = np.array(self.Chan_error_plot)
            plt.hist(Chan_error_array, density =True, cumulative=True, histtype='step', bins=1000)
            plt.title("Chan_Error_CDF", fontsize=22)
            plt.show()
        elif self.Algorithm_comboBox.currentText() == "Chan & Taylor":
            Taylor_error_array = np.array(self.Taylor_error_plot)
            plt.hist(Taylor_error_array, density=True, cumulative=True, histtype='step', bins=1000)
            plt.title("Chan & Taylor_Error_CDF", fontsize=22)
            plt.show()

        pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon('./ECNU.png'))
    mainWindow = QMainWindow()
    ui = Ui_Form()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())


