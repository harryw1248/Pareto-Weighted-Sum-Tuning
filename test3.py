import tkinter as tk
from tkinter import *

root=tk.Tk()
root.title("GridAssign")


Grids=('Grid1','Grid2','Grid3','Grid4','Grid5')
ing=['1','2','3','4','5']
print (len(ing))

variable_dict = {}

for i in range(len(ing)):
    variable = StringVar(root)
    variable.set(Grids[0])
    variable_dict[i] = variable

    label = tk.Label(root,text=ing[i])
    label.grid(row=i,pady=3,sticky=W)
    w=OptionMenu (root, variable, *tuple(Grids))
    w.grid(row=i,column=2,pady=3,sticky=E)
    print(i)
    print(variable.get())

root.mainloop()