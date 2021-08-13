// python.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<Python.h>

int _tmain(int argc, _TCHAR* argv[])
{	
	//Py_Initialize(); /*初始化python解释器,告诉编译器要用的python编译器*/
	//PyRun_SimpleString("import sys"); /*调用python文件*/
	//PyRun_SimpleString("sys.path.append('./')");
    //PyRun_SimpleString("import helloworld"); /*调用python文件*/
    //PyRun_SimpleString("helloworld.printHello()");/*调用python文件中的函数*/
    //Py_Finalize(); /*结束python解释器，释放资源*/

	//根据环境设置pythonHome
	Py_SetPythonHome(L"D:\\ProgramData\\Miniconda3_32");
	Py_Initialize();//使用python之前，要调用Py_Initialize();这个函数进行初始化
	PyRun_SimpleString("import sys"); /*调用python文件*/
	PyRun_SimpleString("sys.path.append('./')");
	PyObject * pModule = NULL;//声明变量
	PyObject * pFunc = NULL;// 声明变量
	pModule = PyImport_ImportModule("chezhe");//这里是要调用的文件名
	pFunc= PyObject_GetAttrString(pModule, "get_RD");//这里是要调用的函数名


	double f[17] = {0, 5, -5, -15, -15, -12.47, -2.48, 7.5, 10, 8, 0, -7.96, -10, -10, -2.48, 5, 0};

	PyObject* pyParams = PyList_New(0);           //初始化一个列表
	for (int i = 0; i < 17; i++)

    {
        PyList_Append(pyParams, Py_BuildValue("d", f[i]));//列表添加元素值浮点数
    }

    PyObject* args = PyTuple_New(1);              //定义一个python变量
    PyTuple_SetItem(args, 0, pyParams);			  // 变量格式转换成python格式

	PyObject* pRetVal = PyEval_CallObject(pFunc, args);//调用函数
	//PyArg_Parse取单个返回值
	int kind=0;
	double RD1 = 0;
	double RD2 = 0;
	PyArg_ParseTuple(pRetVal,"idd",&kind,&RD1, &RD2);

	Py_Finalize();//调用Py_Finalize，这个根Py_Initialize相对应的。

    system("pause");
	return 0;
}





