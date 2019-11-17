from tkinter import *
import tkinter as ttk

# implement error checking later
def generate_menu(pairs):
    num_pairs = len(pairs)
    root = Tk()
    root.title("User Feedback Menu")
    tkvars = []

    # Add a grid
    mainframe = Frame(root)
    mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
    mainframe.columnconfigure(0, weight = 1)
    mainframe.rowconfigure(0, weight = 1)
    mainframe.pack(pady = 100, padx = 100)

    # Create a Tkinter variable
    for i in range(num_pairs):
        tkvar = StringVar(root)
        tkvars.append(tkvar)
    
    choices = set()

    # Dictionary with options
    for pair in pairs:
        choices.add(str(pair))

    row_counter = 1
    for i in range(num_pairs):
        popupMenu = OptionMenu(mainframe, tkvars[i], *choices)
        displayed_text = str(i+1) + "th preferred objective value pair"
        Label(mainframe, text=displayed_text).grid(row = row_counter, column = 1)
        row_counter += 1
        popupMenu.grid(row = row_counter, column =1)
        row_counter += 1

        # on change dropdown value
    def change_dropdown(*args):
        for i in range(num_pairs):
            print(tkvars[i].get())

    # link function to change dropdown
    for i in range(num_pairs):
        tkvars[i].trace('w', change_dropdown)

    
    #tkvars[0].trace('w', change_dropdown)
    #tkvars[1].trace('w', change_dropdown)
    #tkvars[2].trace('w', change_dropdown)
    #tkvars[3].trace('w', change_dropdown)
    #tkvars[4].trace('w', change_dropdown)
    
    print("reached before mainloop")
    root.mainloop()
