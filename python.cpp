// python.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include<Python.h>

int _tmain(int argc, _TCHAR* argv[])
{	
	//Py_Initialize(); /*��ʼ��python������,���߱�����Ҫ�õ�python������*/
	//PyRun_SimpleString("import sys"); /*����python�ļ�*/
	//PyRun_SimpleString("sys.path.append('./')");
    //PyRun_SimpleString("import helloworld"); /*����python�ļ�*/
    //PyRun_SimpleString("helloworld.printHello()");/*����python�ļ��еĺ���*/
    //Py_Finalize(); /*����python���������ͷ���Դ*/

	//���ݻ�������pythonHome
	Py_SetPythonHome(L"D:\\ProgramData\\Miniconda3_32");
	Py_Initialize();//ʹ��python֮ǰ��Ҫ����Py_Initialize();����������г�ʼ��
	PyRun_SimpleString("import sys"); /*����python�ļ�*/
	PyRun_SimpleString("sys.path.append('./')");
	PyObject * pModule = NULL;//��������
	PyObject * pFunc = NULL;// ��������
	pModule = PyImport_ImportModule("chezhe");//������Ҫ���õ��ļ���
	pFunc= PyObject_GetAttrString(pModule, "get_RD");//������Ҫ���õĺ�����


	double f[17] = {0, 5, -5, -15, -15, -12.47, -2.48, 7.5, 10, 8, 0, -7.96, -10, -10, -2.48, 5, 0};

	PyObject* pyParams = PyList_New(0);           //��ʼ��һ���б�
	for (int i = 0; i < 17; i++)

    {
        PyList_Append(pyParams, Py_BuildValue("d", f[i]));//�б����Ԫ��ֵ������
    }

    PyObject* args = PyTuple_New(1);              //����һ��python����
    PyTuple_SetItem(args, 0, pyParams);			  // ������ʽת����python��ʽ

	PyObject* pRetVal = PyEval_CallObject(pFunc, args);//���ú���
	//PyArg_Parseȡ��������ֵ
	int kind=0;
	double RD1 = 0;
	double RD2 = 0;
	PyArg_ParseTuple(pRetVal,"idd",&kind,&RD1, &RD2);

	Py_Finalize();//����Py_Finalize�������Py_Initialize���Ӧ�ġ�

    system("pause");
	return 0;
}





