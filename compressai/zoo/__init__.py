# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from compressai.models import SymmetricalTransFormer, WACNN

#### added ####
from compressai.models import SymmetricalTransFormer_cswin_simple
from compressai.models import SymmetricalTransFormer_cswin_general
from compressai.models import SymmetricalTransFormer_cswin_general_321
from compressai.models import SymmetricalTransFormer_cswin_general_321_embed_742
from compressai.models import SymmetricalTransFormer_cswin_general_embed_742
from compressai.models import SymmetricalTransFormer_cswin_simple_RPE
from compressai.models import SymmetricalTransFormer_cswin_general_window_2_2_2_2
from compressai.models import SymmetricalTransFormer_cswin_general_window_4_4_4_4
from .pretrained import load_pretrained as load_state_dict

models = {
    'stf': SymmetricalTransFormer,
    'cstf_simple': SymmetricalTransFormer_cswin_simple,
    'cstf_general': SymmetricalTransFormer_cswin_general,
    'cstf_general_321' : SymmetricalTransFormer_cswin_general_321,
    'cstf_general_321_embed_742' : SymmetricalTransFormer_cswin_general_321_embed_742,
    'cstf_general_embed_742' : SymmetricalTransFormer_cswin_general_embed_742,
    'cstf_simple_RPE' : SymmetricalTransFormer_cswin_simple_RPE,
    'cstf_general_window_2_2_2_2' : SymmetricalTransFormer_cswin_general_window_2_2_2_2,
    'cstf_general_window_4_4_4_4' : SymmetricalTransFormer_cswin_general_window_4_4_4_4,
    'cnn': WACNN,
    
}
