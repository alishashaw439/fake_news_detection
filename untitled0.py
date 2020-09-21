animals=[
    ["a","b","c"],
    ["r","f","s"],
    ["w","p","t"]
    
]
class Student:
    
    
    def __init__(self,name,age,gpa):
        self.name=name
        self.age=age
        self.gpa=gpa
        
    def __init__(self,school,year):
        self.school=school
        self.year=year

student1=Student("alisha",21,8.04)
student2=Student("thomas",2)
print(student2.year)




