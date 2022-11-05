import tkinter
import time

class ConvertNotice(tkinter.Toplevel):

    def __init__(self, file_name):
        super().__init__()
        self.title("通知")
        self.geometry("400x60+700+400")
        self.file_name = file_name

        # 弹窗界面
        self.row1 = tkinter.Frame(self)
        self.row1.pack(fill="x")
        tkinter.Label(self.row1, text="正在将 {}.doc 转换为PDF中，请稍等！".format(self.file_name), font=('微软雅黑', 9), width=50).pack()

    def success(self):
        tkinter.Label(self.row1, text="{}.doc 转换成功！".format(self.file_name), font=('微软雅黑', 9), width=50).pack()
        time.sleep(2)

    def close(self):
        self.destroy()



notice = ConvertNotice("dd")
notice.success()