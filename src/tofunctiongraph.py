#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:09:14 2020

@author: michael
"""

import onnx
from onnx import numpy_helper
import networkx as nx
import matplotlib.pyplot as plt

functions = {}


def to_graph(filename, keep_connections_nodes=False):
     
    model = onnx.load(filename)
     
    inits = model.graph.initializer

    weights = {}
    
    connection_nodes = {}
    
    graph = N4JGraph("bolt://localhost:7687/", "neo4j", "password")#nx.DiGraph()
    
    in_name_list = []

    for i in inits:
        weights[i.name] = numpy_helper.to_array(i)


    conv_nodes = []
    dense_nodes = []
    one_to_one_nodes = []
    reshape_nodes = []
    for n in model.graph.node:
        f_name = n.op_type
        if f_name == 'Conv':
            conv_nodes.append(n)
        elif f_name in ['Gemm']:
            dense_nodes.append(n)
        elif f_name in ['Relu', 'LeakyRelu', 'ThresholdRelu', 'PRelu', 'Tanh',
                      'Sigmoid', 'Affine', 'ScaledTanh', 'HardSigmoid', 'Elu',
                      'Softsign', 'Softplus', 'Abs', 'Acos', 'Acosh', 'Asin',
                      'Asinh', 'Atan', 'Atanh', 'Ceil', 'Cos', 'Cosh', 'Floor',
                      'Sin', 'Sinh', 'Sqrt', 'Tan', 'Clip', 'Round', 'Pow', 
                      'Neg', 'Mod', 'AveragePool', 'BatchNormalization',
                      'BitShift', 'Cast', 'Dropout']:
            one_to_one_nodes.append(n)
        elif f_name in ['Reshape', 'Transpose']:
            reshape_nodes.append(n)
        elif f_name == 'Constant':
            pass
        else:
            print(n.op_type)
        
    #Functions with a very specific number of inputs and outputs
    for n in conv_nodes:
        f_name = n.op_type
        att_list = n.attribute
        attributes = {}
        inputs = n.input
        for a in att_list:
            attributes[a.name] = a
        name = '_'
        
        #Break a convolution up into little pieces
        #generate name for this specific type of convolution
        name = "_Conv"
        name = name + '_ks:' + str(attributes['kernel_shape'].ints)
        if 'dilations' in att_list:
            name = name + '_d:' + str(attributes['dilations'].ints)
        
        name = name + '_s:' + str(attributes['strides'].ints)
        name = name + '_p:' + str(attributes['pads'].ints)
        if 'group' in att_list:
            name = name + '_g:' + str(attributes['group'].i)
        print(name)
        #print(n)
        #now, to split up the channels of the convolution
        shape = attributes['kernel_shape']
        weight_name = None
        bias_name = None
        in_name = inputs[0]
        weight_name = inputs[1]
        bias_name = None 
        if len(inputs) > 2: bias_name = inputs[2]
        out_name = n.output[0]
        
        w = weights[weight_name]
        b = None
            
        if bias_name is not None: b = weights[bias_name]
            
        #Create image in nodes for each in channel
        in_list = []
        if in_name not in connection_nodes.keys():
            in_list = []
            for channel in range(w.shape[1]):
                new_node_name = in_name + '_' + str(channel)
                #graph.add_node(new_node_name, channel=channel, typ="img")
                graph.add_node("Image", {'name':new_node_name, 'channel':channel})
                in_list.append(new_node_name)
            connection_nodes[in_name] = in_list
        else:
            in_list = connection_nodes[in_name]
            
        #Create image out nodes for each out channel
        out_list = []
        if out_name not in connection_nodes.keys():
            out_list = []
            for channel in range(w.shape[0]):
                new_node_name = out_name + '_' + str(channel)
                graph.add_node("Image", {"name":new_node_name, "group":in_name, "channel":channel})
                out_list.append(new_node_name)
            connection_nodes[out_name] = out_list
            
        #Create covolution nodes for each kernel matrix
        for out_channel in range(w.shape[0]):
            
            func_nodes = []
            
            for in_channel in range(w.shape[1]):
                
                kernel = w[out_channel, in_channel, :, :]
                flat_kernel = kernel.flatten()
                new_func_node = name + '_' + in_name + '_' + str(in_channel) + '_' +str(out_channel)
                func_nodes.append(new_func_node)
                func_id = graph.add_node("Conv", {"name":new_func_node,
                                        "func_name":name,
                                        "weights":flat_kernel,
                                        "ic":in_channel,
                                        "oc":out_channel,
                                        "ina":in_name})
                in_tensor_id = graph.get_id_from_name(in_name + '_' + str(in_channel))
                
                graph.add_edge(in_tensor_id, func_id, "TO", {})
                inter_id = graph.add_node("Image", {'name':'inter_'+new_func_node, "ic":in_channel, "oc":out_channel, "ina":in_name})
                graph.add_edge(func_id, inter_id, "TO", {})
            
            new_sum_node = 'convsum_' + in_name + '_' + str(out_channel)
            bi = 0
            if b is not None: bi = b[out_channel]
            wis = graph.add_node("Weighted_Imgsum", {"name":new_sum_node, "func_name":"weighted_imgsum", "weights":[bi]})
            for f in func_nodes:
                graph.add_edge(graph.get_id_from_name(f), wis, "TO", {})
            graph.add_edge(wis, graph.get_id_from_name(out_name + '_' + str(out_channel)), "TO", {})
            
    
    for n in dense_nodes:
        f_name = n.op_type
        att_list = n.attribute
        attributes = {}
        inputs = n.input
        in_name = inputs[0]
        weight_name = inputs[1]
        bias_name = inputs[2]
        out_name = n.output[0]
        
        w = weights[weight_name]
        b = weights[bias_name]
        n_outs = w.shape[0]
        n_ins = w.shape[1]
        
        in_list = []
        if in_name not in connection_nodes.keys():
            in_list = []
            for channel in range(w.shape[1]):
                new_node_name = in_name + '_' + str(channel)
                graph.add_node("Float", {"name":new_node_name, "channel":channel})
                in_list.append(new_node_name)
            connection_nodes[in_name] = in_list
        else:
            in_list = connection_nodes[in_name]
        
        outlist = []
        if out_name not in connection_nodes.keys():
            out_list = []
            for channel in range(w.shape[0]):
                new_node_name = out_name + '_' + str(channel)
                graph.add_node("Float", {"name":new_node_name, "channel":channel})
                out_list.append(new_node_name)
            connection_nodes[out_name] = out_list
        else:
            out_list = connection_nodes[out_name]
            
        for o in range(len(out_list)):
            onn = out_list[o]
            
            bias_node_name = "_mm_biased_sum_" + str(o)
            bsid = graph.add_node("Bias", {"name":bias_node_name, "weights":b[o]})
            graph.add_edge(graph.get_id_from_name(onn), bsid, "TO", {})
            
            for i in range(len(in_list)):
                inn = in_list[i]
                mm_node_name = "_mm_mult_" + str(o) + "_" + str(i)
                mmn = graph.add_node("MatMul", {"name":mm_node_name, "weights":w[o,i]})
                graph.add_edge(graph.get_id_from_name(inn), mmn, "TO", {})
                graph.add_edge(mmn, bsid, "TO", {})
                
            #print(graph.size())
                
            
        print(n)
            
            
            
            
            
        
    
    #One-to-one functions
    ni = 0
    while one_to_one_nodes:
        n = one_to_one_nodes[ni]
        f_name = n.op_type
        att_list = n.attribute
        attributes = {}
        inputs = n.input
        
        for a in att_list:
            attributes[a.name] = a
            
        name = f_name
        in_name = inputs[0]
        out_name = n.output[0]
        
        w = []
        weight_names=inputs[1:]
        for wn in weight_names:
            we = weights[wn].flatten()
            w = w + we
        
        
        if not (in_name not in connection_nodes.keys() and out_name not in connection_nodes.keys()):
            
            #if we only have the inputs or outputs, generate the outputs or inputs
            if out_name not in connection_nodes.keys():
                connection_nodes[out_name] = []
                for inn in connection_nodes[in_name]:
                    nnn = out_name + inn[len(in_name):]
                    connection_nodes[out_name].append(nnn)
                    graph.add_node("Placeholder", {"name":nnn})
            elif in_name not in connection_nodes.keys():
                connection_nodes[in_name] = []
                for onn in connection_nodes[out_name]:
                    nnn = in_name + onn[len(out_name):]
                    connection_nodes[in_name].append(nnn)
                    graph.add_node("Placeholder", {"name":nnn})
            
            for inn in connection_nodes[in_name]:
                onn = out_name + inn[len(in_name):]
                nn_name = name+'_'+in_name+'_'+out_name+inn[len(in_name):]
                nn = graph.add_node(f_name, {"name":nn_name, "func_name":f_name, "weights":w})
                graph.add_edge(graph.get_id_from_name(inn), nn, "TO", {})
                graph.add_edge(nn, graph.get_id_by_name(onn), "TO", {})
        
            one_to_one_nodes.remove(n)
            
        else:
            ni = (ni+1)%len(one_to_one_nodes)

    ni = 0
    while reshape_nodes:
        n = reshape_nodes[ni]
        
        f_name = n.op_type
        att_list = n.attribute
        attributes = {}
        inputs = n.input
        
        for a in att_list:
            attributes[a.name] = a
        
        if not (in_name not in connection_nodes.keys() and out_name not in connection_nodes.keys()):
            
            print(f_name)
            print(att_list)
            
            reshape_nodes.remove(n)
            
        else:
            
            ni = (ni+1)%len(one_to_one_nodes)
            
            
            
    #if keep_connections_nodes == False:
    #    for k in connection_nodes.keys():
    #        for c in connection_nodes[k]:
    #            for inn in graph.predecessors(c):
    #                for onn in graph.successors(c):
    #                    graph.add_edge(inn, onn)
    #            graph.remove_node(c)
            
    #graph.name = filename
    return graph
        
        

G = to_graph('modelzoo/onnx/alexnet.onnx', keep_connections_nodes=False)

nx.write_gpickle(G, "test.xnn")

#print(G)

#pos = nx.spring_layout(G)
#nx.draw_networkx_nodes(G, pos,
#                       node_color='r',
#                       node_size=500,
#                       alpha=0.8)
#nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
#plt.show()