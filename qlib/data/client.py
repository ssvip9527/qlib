# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import socketio

import qlib
from ..config import C
from ..log import get_module_logger
import pickle


class Client:
    """客户端类

    为ClientProvider提供连接工具函数。
    """

    def __init__(self, host, port):
        super(Client, self).__init__()
        self.sio = socketio.Client()
        self.server_host = host
        self.server_port = port
        self.logger = get_module_logger(self.__class__.__name__)
        # bind connect/disconnect callbacks
        self.sio.on(
            "connect",
            lambda: self.logger.debug("Connect to server {}".format(self.sio.connection_url)),
        )
        self.sio.on("disconnect", lambda: self.logger.debug("从服务器断开连接!"))

    def connect_server(self):
        """连接到服务器。"""
        try:
            self.sio.connect(f"ws://{self.server_host}:{self.server_port}")
        except socketio.exceptions.ConnectionError:
            self.logger.error("无法连接到服务器 - 请检查您的网络或服务器状态")

    def disconnect(self):
        """从服务器断开连接。"""
        try:
            self.sio.eio.disconnect(True)
        except Exception as e:
            self.logger.error("无法从服务器断开连接: %s" % e)

    def send_request(self, request_type, request_content, msg_queue, msg_proc_func=None):
        """向服务器发送特定请求。

        参数
        ----------
        request_type : str
            请求的类型，'calendar'/'instrument'/'feature'。
        request_content : dict
            记录请求的信息。
        msg_proc_func : func
            接收响应时处理消息的函数，应包含参数`*args`。
        msg_queue: Queue
            回调后传递消息的队列。
        """
        head_info = {"version": qlib.__version__}

        def request_callback(*args):
            """回调包装器

            :param *args: args[0]是响应内容
            """
            # args[0] is the response content
            self.logger.debug("接收数据并进入队列")
            msg = dict(args[0])
            if msg["detailed_info"] is not None:
                if msg["status"] != 0:
                    self.logger.error(msg["detailed_info"])
                else:
                    self.logger.info(msg["detailed_info"])
            if msg["status"] != 0:
                ex = ValueError(f"响应错误(status=={msg['status']}), 详细信息: {msg['detailed_info']}")
                msg_queue.put(ex)
            else:
                if msg_proc_func is not None:
                    try:
                        ret = msg_proc_func(msg["result"])
                    except Exception as e:
                        self.logger.exception("处理消息时出错。")
                        ret = e
                else:
                    ret = msg["result"]
                msg_queue.put(ret)
            self.disconnect()
            self.logger.debug("已断开连接")

        self.logger.debug("尝试连接")
        self.connect_server()
        self.logger.debug("已连接")
        # pickle用于传递一些特殊类型的参数（如pd.Timestamp）
        request_content = {"head": head_info, "body": pickle.dumps(request_content, protocol=C.dump_protocol_version)}
        self.sio.on(request_type + "_response", request_callback)
        self.logger.debug("尝试发送")
        self.sio.emit(request_type + "_request", request_content)
        self.sio.wait()
