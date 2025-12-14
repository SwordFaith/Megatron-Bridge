# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from megatron.bridge.models.conversion.param_mapping import (
    ChunkedMapping,
    GDNConv1dMapping,
    MambaConv1dMapping,
)


class TestMappingInheritanceRegression:
    """
    Regression tests to ensure mappings inherit from the correct base classes
    and implement required abstract methods.
    """

    def test_mamba_conv1d_mapping_inheritance(self):
        """
        Regression test: Verify MambaConv1dMapping inherits from ChunkedMapping.
        
        Issue context: Previously MambaConv1dMapping inherited directly from
        MegatronParamMapping, missing the shared hf_to_megatron/megatron_to_hf
        implementation provided by ChunkedMapping.
        """
        assert issubclass(MambaConv1dMapping, ChunkedMapping), \
            "MambaConv1dMapping must inherit from ChunkedMapping to reuse shared split/merge logic"

    def test_gdn_conv1d_mapping_inheritance(self):
        """Verify GDNConv1dMapping inherits from ChunkedMapping."""
        assert issubclass(GDNConv1dMapping, ChunkedMapping), \
            "GDNConv1dMapping must inherit from ChunkedMapping"

    def test_mamba_conv1d_mapping_instantiation(self):
        """
        Verify MambaConv1dMapping can be instantiated (abstract methods implemented).
        If inheritance is wrong, this raises TypeError.
        """
        try:
            # Instantiate with dummy names
            mapping = MambaConv1dMapping("mamba.conv1d.weight", "hf.conv1d.weight")
        except TypeError as e:
            pytest.fail(f"Could not instantiate MambaConv1dMapping. Likely missing abstract methods: {e}")
        
        assert hasattr(mapping, "hf_to_megatron")
        assert hasattr(mapping, "megatron_to_hf")

    def test_gdn_conv1d_mapping_instantiation(self):
        """Verify GDNConv1dMapping can be instantiated."""
        try:
            # Instantiate with dummy names
            mapping = GDNConv1dMapping("gdn.conv1d.weight", "hf.conv1d.weight")
        except TypeError as e:
            pytest.fail(f"Could not instantiate GDNConv1dMapping. Likely missing abstract methods: {e}")

        assert hasattr(mapping, "hf_to_megatron")
        assert hasattr(mapping, "megatron_to_hf")
