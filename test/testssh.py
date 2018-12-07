import paramiko
import time


#paramiko.util.log_to_file('/tmp/sshout')
def ssh2(ip, username, passwd, cmd):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, 22, username, passwd, timeout=5)
        stdin, stdout, stderr = ssh.exec_command(cmd)
        #           stdin.write("Y")   #简单交互，输入 ‘Y’
        print (stdout.read())
        #        for x in  stdout.readlines():
        #          print x.strip("n")
        print ('%stOKn' % (ip))
        ssh.close()
    except:
        print ('%stErrorn' % (ip))


ssh2("140.113.*******", "pi", "830725", "cd AIY-projects-python  \n env/bin/python checkpoints/check_audio.py")
