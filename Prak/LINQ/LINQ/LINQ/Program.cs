using System;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Drawing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Xml;
using System.Xml.Linq;
using Newtonsoft.Json;
using static System.Net.Mime.MediaTypeNames;
using static H;

class H
{
    public class task
    {
        public string taskName;
        public student[] data;
    }
    public class student
    {
        public string name;
        public string group;
        public string discipline;
        public int mark;
    }


    public class answ1
    {
        public Response1[] Response;
    }
    public class Response1
    {
        public string Cadet;
        public int GPA;
    }



    public class answ2
    {
        public string name;
        public student[] students;
    }



    public class answ3
    {
        public string name;
        public student[] students;
    }





    static void Main()
    {
        Task("input1.txt", "output1.txt");
        Task("input2.txt", "output2.txt");
        Task("input3.txt", "output3.txt");
    } 
    static void Task(string input_file, string output_file)
    {
        StreamReader sr0 = new StreamReader(input_file);
        string json=sr0.ReadToEnd();
        task m = JsonConvert.DeserializeObject<task>(json);
        if (m.taskName== "GetStudentsWithHighestGPA")
        {
            Task1(m, output_file);
        }
        else if (m.taskName == "GetBestGroupsByDiscipline")
        {
            Task2(m, output_file);
        }
        else if (m.taskName == "CalculateGPAByDiscipline")
        {
            Task3(m, output_file);
        }
        else
        {
            Console.WriteLine("Неизвестное задание");
        }
        sr0.Close();
    }

    // Определить студента/студентов с максимальным средним баллом. 
    static void Task1(task m, string output_file)
    {
        var group_by_student = from i in m.data
                               group i by i.name;




        IEnumerable<Response1> answer;

        Console.WriteLine("Неизвестное задание");
    }
    static void Task2(task m, string output_file)
    {

    }
    static void Task3(task m, string output_file)
    {

    }

    static void Answ(string output_file, task m)
    {
        StreamWriter sr1 = new StreamWriter(output_file);
        string json = JsonConvert.SerializeObject(m);
        sr1.Write(json);
        sr1.Close();
    }
}

    