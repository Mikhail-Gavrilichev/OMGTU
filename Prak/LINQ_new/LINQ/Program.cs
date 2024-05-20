using System;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Drawing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
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
        public string? Cadet;
        public double GPA;
    }



    public class answ2
    {
        public Response2[] Response;
    }
    public class Response2
    {
        public string? discipline;
        public double GPA;
    }

    
    public class answ3
    {
        public Response3[] Response;
    }
    public class Response3
    {
        public string? Discipline;  
        public string? Group;
        public double GPA;
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
        else if (m.taskName == "CalculateGPAByDiscipline")
        {
            Task2(m, output_file);
        }
        else if (m.taskName == "GetBestGroupsByDiscipline")
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

        List<Response1> students = new List<Response1>();
        foreach (var group in group_by_student)
        {
            Response1 student = new Response1();
            student.Cadet = group.Key;
            student.GPA   = Convert.ToDouble(group.Sum(a => a.mark)) / Convert.ToDouble(group.Count());
            students.Add(student);
        }

        List<Response1> student_with_max_GPA = new List<Response1>();
        double max = students.Max(a => a.GPA);
        foreach (var student in students)
        {
            if (student.GPA == max)
                student_with_max_GPA.Add(student);
        }

        answ1 answ = new answ1();
        answ.Response = student_with_max_GPA.ToArray();

        string json = JsonConvert.SerializeObject(answ);

        StreamWriter sw = new StreamWriter(output_file);
        sw.Write(json);
        sw.Close();
    }
    // Вычислить средний бал по каждому предмету
    static void Task2(task m, string output_file)
    {
        var group_by_discipline = from i in m.data
                                  group i by i.discipline;

        List<Response2> students = new List<Response2>();
        foreach (var group in group_by_discipline)
        {
            Response2 student = new Response2();
            student.discipline = group.Key;
            student.GPA = Convert.ToDouble(group.Sum(a => a.mark)) / Convert.ToDouble(group.Count());
            students.Add(student);
        }

        answ2 answ = new answ2();
        answ.Response = students.ToArray();

        string json = JsonConvert.SerializeObject(answ);

        StreamWriter sw = new StreamWriter(output_file);
        sw.Write(json);
        sw.Close();
    }
 
    //По каждому предмету определить группу с лучшим средним баллом
    static void Task3(task m, string output_file)
    {
        var group_by_discipline = from i in m.data
                                  group i by i.discipline;

        List<Response3> resp = new List<Response3>();
        foreach (var group in group_by_discipline)
        {
            var group_by_dgroup = from i in @group
                                  group i by i.@group;

            List<Response3> gap_by_discipline = new List<Response3>();
            foreach (var disc in group_by_dgroup)
            {
                Response3 one_disc = new Response3();
                one_disc.Group = disc.Key;
                one_disc.GPA = Convert.ToDouble(disc.Sum(a => a.mark)) / Convert.ToDouble(disc.Count());
                gap_by_discipline.Add(one_disc);
            }

            
            if (gap_by_discipline.Count() > 0)
            {
                double max = gap_by_discipline.Max(a => a.GPA);
                foreach (var gr in gap_by_discipline)
                {
                    if(max == gr.GPA)
                    {
                        gr.Discipline = group.Key;
                        resp.Add(gr);
                    }
                }
            }
        }

        answ3 answ = new answ3();
        answ.Response = resp.ToArray();

        string json = JsonConvert.SerializeObject(answ);

        StreamWriter sw = new StreamWriter(output_file);
        sw.Write(json);
        sw.Close();
    }
}

    