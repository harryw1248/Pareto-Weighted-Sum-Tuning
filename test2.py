from tkinter import *
import tkinter as ttk
#from ttk import *

root = Tk()
root.title("Tk dropdown example")

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.pack(pady = 100, padx = 100)

# Create a Tkinter variable
tkvar1 = StringVar(root)
tkvar2 = StringVar(root)

# Dictionary with options
choices = { 'Pizza','Lasagne','Fries','Fish','Potatoe'}
tkvar1.set('Pizza') # set the default option
tkvar2.set('Pizza') # set the default option

popupMenu = OptionMenu(mainframe, tkvar1, *choices)
Label(mainframe, text="Choose a dish").grid(row = 1, column = 1)
popupMenu.grid(row = 2, column =1)

popupMenu2 = OptionMenu(mainframe, tkvar2, *choices)
Label(mainframe, text="Choose a dish").grid(row = 3, column = 1)
popupMenu2.grid(row = 4, column =1)

# on change dropdown value
def change_dropdown(tk):
    print( tk.get() )

# link function to change dropdown
tkvar1.trace('w', change_dropdown(tkvar1))
tkvar2.trace('w', change_dropdown(tkvar2))

root.mainloop()