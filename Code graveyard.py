#for k in KB:
# # Identify depots
#     depots = [node.id for node in nodes.values() if node.type == 'depot']
#     for depot in depots:
#         # Each barge can depart from a depot to at most one outgoing arc
#         model.addConstr(
#             quicksum(x_ijk[k][(depot, j)] for j in N if j != depot and (depot, j) in x_ijk[k]) <= 1,
#             name=f"Depart_{k}_{depot}"
#         )
#     for i in N:
#         if i not in depots:
#             # For non-depot nodes, ensure flow conservation
#             model.addConstr(
#                 quicksum(x_ijk[k][(i, j)] for j in N if j != i and (i, j) in x_ijk[k]) -
#                 quicksum(x_ijk[k][(j, i)] for j in N if j != i and (j, i) in x_ijk[k]) == 0,
#                 name=f"Flow_{i}_{k}"
#             )
#             # Explanation:
#             # Ensures that barges entering a node also leave it, maintaining a continuous route


#(3)

#for k in KB:
# for j in N:
#         if nodes[j].type == 'terminal':
#             model.addConstr(
#                 p_jk[j, k] == quicksum(
#                     Wc[c] * Zcj.get((c, j), 0) * f_ck[c, k]
#                     for c in I
#                 ),
#                 name=f"Import_{j}_{k}"
#             )
#             # Explanation:
#             # Defines the total import quantity loaded by barge k at terminal j
#             # Sum of sizes of all import containers assigned to barge k originating from j


# # (4) Export quantities unloaded by barge k at sea terminal j
# for k in KB:
#     for j in N:
#         if nodes[j].type == 'terminal':
#             model.addConstr(
#                 d_jk[j, k] == quicksum(
#                     Wc[c] * Zcj.get((c, j), 0) * f_ck[c, k]
#                     for c in E
#                 ),
#                 name=f"Export_{j}_{k}"
#             )
#             # Explanation:
#             # Defines the total export quantity unloaded by barge k at terminal j
#             # Sum of sizes of all export containers assigned to barge k destined to j


# quicksum(
#     x_ijk[k][(i, j)] * (HBk[k] if i in [node.id for node in nodes.values() if node.type == 'depot'] else 0)
#     for k in KB for (i, j) in x_ijk[k]
# )

# quicksum(
#             Tij[i, j] * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k]
#         ) +


# quicksum(
#     gamma * x_ijk[k][(i, j)] for k in KB for (i, j) in x_ijk[k] if nodes[j].type == 'terminal'
# ),
