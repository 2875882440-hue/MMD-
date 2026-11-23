"""Small labelled engineering fixture, never a Rin asset binding."""
from motionbridge.secondary_segmentation import build_inventory,review_gate,_sha


def make_fixture(mode='surface_rigid'):
    positions=[[-.1,-.1,0],[.1,-.1,0],[0,.1,0],[-.1,-.1,.02],[.1,-.1,.02],[0,.1,.02],
               [.4,0,0],[.5,0,0],[.4,.1,0],[1,0,0],[1.1,0,0],[1,.1,0]]
    plan={'model_sha256':'synthetic-engineering-fixture','protected_bone_ids':[0],'bones':[{'name':'fixture_root'}],
      'geometry':{'positions_m':positions,'triangles':[[0,1,2],[3,4,5],[6,7,8],[9,10,11]],
       'face_material_ids':[0,1,0,2],'material_names':['fixture_cloth','fixture_ornament','fixture_protected'],
       'skin_weights':[{'0':1.} for _ in positions]},
      'regions':[{'role':'cloth','face_ids':[0,2],'vertex_ids':[0,1,2,6,7,8]},
       {'role':'accessory','face_ids':[1],'vertex_ids':[3,4,5]},
       {'role':'body','face_ids':[3],'vertex_ids':[9,10,11]}]}
    inventory=build_inventory(plan)
    review={'schema_version':1,'model_sha256':plan['model_sha256'],'geometry_sha256':inventory['geometry_sha256'],
      'parts':[{'id':'cloth','category':'outer_cloth','layer':1,'face_ids':[0,2],'evidence':'declared engineering cloth'},
       {'id':'ornament','category':'flexible_trim' if mode=='surface_seam' else 'rigid_ornament','face_ids':[1],'evidence':'declared engineering ornament'},
       {'id':'protected','category':'protected_surface','face_ids':[3],'evidence':'immutable source fixture'}], 'overlaps':[]}
    plan['surface_segmentation']={'inventory':inventory,'review_gate':review_gate(inventory,review),'required_for_surface_dynamics':True}
    row={'part_id':'ornament','mode':mode,'evidence':'fixture explicit anchors',
         'limits':{'translation_m':.25,'rotation_rad':.8}}
    if mode=='bone_socket':
        row.update(bone_id=0,limits={'translation_m':0,'rotation_rad':0},anchors=[{'vertex_id':v} for v in (3,4,5)])
    else:
        row.update(target_part='cloth',anchors=[{'vertex_id':i+3,'target_face_id':0,
            'barycentric':[int(i==j) for j in range(3)],'rest_offset_m':[0,0,.02]} for i in range(3)])
    bindings={'schema_version':1,'model_sha256':plan['model_sha256'],'geometry_sha256':inventory['geometry_sha256'],
      'review_sha256':_sha(review),'attachments':[
       {'part_id':'cloth','mode':'source_root','anchors':[{'vertex_id':6}],
        'limits':{'translation_m':0,'rotation_rad':0},'evidence':'fixture source root'},row], 'aliases':[],'seams':[]}
    return plan,review,bindings
