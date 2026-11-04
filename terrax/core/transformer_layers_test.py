# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
import numpy as np
from terrax.core import boundaries
from terrax.core import coordinates
from terrax.core import spherical_harmonics
from terrax.core import standard_layers
from terrax.core import transformer_layers


class TransformerLayersTest(parameterized.TestCase):
  """Tests transformer layers instantiate and produce expected outputs."""

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    self.levels = coordinates.SigmaLevels.equidistant(12)
    self.hidden_size = 8
    self.per_head_kv_size = 3
    self.dense_factory = functools.partial(
        standard_layers.Mlp.uniform,
        hidden_layers=1,
        hidden_size=self.hidden_size,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_gating',
          input_size=6,
          output_size=3,
          num_heads=2,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='gating_where_possible',
          input_size=5,
          output_size=2,
          num_heads=3,
          gating=None,
      ),
  )
  def test_transformer_blocks_self_attention(
      self,
      input_size: int,
      output_size: int,
      num_heads: int,
      gating: transformer_layers.Gating,
  ):
    """Tests output_shape of TransformerBlocks."""
    n_levels = self.levels.shape[0]
    net = transformer_layers.TransformerBlocks.build_using_factories(
        input_size,
        output_size,
        intermediate_sizes=[self.hidden_size, self.hidden_size],
        num_heads=num_heads,
        dense_factory=self.dense_factory,
        qkv_features=(num_heads * self.per_head_kv_size),
        gating=gating,
        rngs=nnx.Rngs(0),
    )
    rng = jax.random.key(0)
    inputs = jax.random.uniform(rng, (input_size, n_levels))
    out = net(inputs)
    self.assertEqual(out.shape, (output_size, n_levels))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_gating',
          output_size=3,
          num_heads=2,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='gating_where_possible',
          output_size=2,
          num_heads=3,
          gating=None,
      ),
      dict(
          testcase_name='with_norm_layer',
          output_size=2,
          num_heads=3,
          gating=None,
          normalization=nnx.LayerNorm,
      ),
      dict(
          testcase_name='with_dynamic_tanh',
          output_size=2,
          num_heads=3,
          gating=None,
          normalization=transformer_layers.DyT,
      ),
  )
  def test_transformer_blocks_decode(
      self,
      output_size: int,
      num_heads: int,
      gating: transformer_layers.Gating,
      normalization: transformer_layers.NormalizationFactory | None = None,
  ):
    """Tests output_shape of TransformerBlocks."""
    n_levels = self.levels.shape[0]
    # NOTE: In this case we apply both pre and post normalization layers,
    # this is not a recommended configuration, but works as an effective test.
    net = transformer_layers.TransformerBlocks.build_using_factories(
        self.hidden_size,
        output_size,
        intermediate_sizes=[self.hidden_size, self.hidden_size],
        num_heads=num_heads,
        dense_factory=self.dense_factory,
        qkv_features=(num_heads * self.per_head_kv_size),
        gating=gating,
        pre_normalization_factory=normalization,
        post_normalization_factory=normalization,
        rngs=nnx.Rngs(0),
    )
    rng = jax.random.key(0)
    inputs = jax.random.uniform(rng, (self.hidden_size, n_levels))
    levels_in_latents = 13
    latents = jax.random.uniform(rng, (self.hidden_size, levels_in_latents))
    out = net(inputs, latents=latents)
    self.assertEqual(out.shape, (output_size, n_levels))

  def test_default_qkv_features_follow_layer_input_sizes(self):
    """Tests that qkv_features default to the input size of each layer."""
    net = transformer_layers.TransformerBlocks.build_using_factories(
        6,
        4,
        intermediate_sizes=[8],
        num_heads=2,
        dense_factory=self.dense_factory,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual([a.qkv_features for a in net.attentions], [6, 8])

  @parameterized.parameters(False, True)
  def test_post_norms_of_last_block(self, apply_final_norm: bool):
    """Tests that apply_final_norm only controls the final post-dense norm."""
    net = transformer_layers.TransformerBlocks.build_using_factories(
        self.hidden_size,
        intermediate_sizes=[self.hidden_size],
        num_heads=2,
        dense_factory=self.dense_factory,
        post_normalization_factory=nnx.LayerNorm,
        apply_final_norm=apply_final_norm,
        rngs=nnx.Rngs(0),
    )
    post_norms = list(net.post_norms)
    self.assertLen(post_norms, 4)  # (post-attention, post-dense) x 2 blocks.
    for norm in post_norms[:3]:
      self.assertIsInstance(norm, nnx.LayerNorm)
    if apply_final_norm:
      self.assertIsInstance(post_norms[3], nnx.LayerNorm)
    else:
      self.assertIs(post_norms[3], transformer_layers.identity)

  def test_transformer_blocks_relative_bias(self):
    """Tests TransformerBlocks with relative position bias."""
    n_levels = self.levels.shape[0]
    num_heads, pos_encoding_size = 2, 5
    relative_bias_factory = functools.partial(
        standard_layers.Mlp, intermediate_sizes=()
    )
    net = transformer_layers.TransformerBlocks.build_using_factories(
        self.hidden_size,
        intermediate_sizes=[self.hidden_size],
        num_heads=num_heads,
        dense_factory=self.dense_factory,
        relative_bias_factory=relative_bias_factory,
        pos_encoding_size=pos_encoding_size,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(net.relative_bias_net.input_size, pos_encoding_size)
    self.assertEqual(net.relative_bias_net.output_size, num_heads)
    inputs = jax.random.uniform(jax.random.key(0), (self.hidden_size, n_levels))
    pe = jax.random.uniform(jax.random.key(1), (pos_encoding_size, n_levels))
    with self.subTest('self_attention'):
      out = net(inputs, inputs_pos_encoding=pe)
      self.assertEqual(out.shape, (self.hidden_size, n_levels))
    with self.subTest('latents_share_inputs_pos_encoding'):
      latents = jax.random.uniform(
          jax.random.key(2), (self.hidden_size, n_levels)
      )
      out = net(inputs, latents=latents, inputs_pos_encoding=pe)
      self.assertEqual(out.shape, (self.hidden_size, n_levels))
    with self.subTest('requires_pos_encoding'):
      with self.assertRaisesRegex(ValueError, 'inputs_pos_encoding'):
        net(inputs)
    with self.subTest('factory_requires_pos_encoding_size'):
      with self.assertRaisesRegex(ValueError, 'pos_encoding_size'):
        transformer_layers.TransformerBlocks.build_using_factories(
            self.hidden_size,
            intermediate_sizes=[],
            num_heads=num_heads,
            dense_factory=self.dense_factory,
            relative_bias_factory=relative_bias_factory,
            rngs=nnx.Rngs(0),
        )

  def test_transformer_blocks_additive_pos_encoding(self):
    """Tests TransformerBlocks with projected positional encodings as inputs."""
    n_levels = self.levels.shape[0]
    pos_encoding_size = 5
    net = transformer_layers.TransformerBlocks.build_using_factories(
        self.hidden_size,
        intermediate_sizes=[self.hidden_size],
        num_heads=2,
        dense_factory=self.dense_factory,
        pos_encoding_size=pos_encoding_size,
        use_additive_pos_encoding=True,
        rngs=nnx.Rngs(0),
    )
    self.assertIsInstance(net.pos_encoding_proj, nnx.Linear)
    self.assertEqual(net.pos_encoding_proj.in_features, pos_encoding_size)
    self.assertEqual(net.pos_encoding_proj.out_features, self.hidden_size)
    inputs = jax.random.uniform(jax.random.key(0), (self.hidden_size, n_levels))
    pe = jax.random.uniform(jax.random.key(1), (pos_encoding_size, n_levels))
    out = net(inputs, inputs_pos_encoding=pe)
    self.assertEqual(out.shape, (self.hidden_size, n_levels))
    with self.subTest('requires_pos_encoding'):
      with self.assertRaisesRegex(ValueError, 'inputs_pos_encoding'):
        net(inputs)
    with self.subTest('factory_requires_pos_encoding_size'):
      with self.assertRaisesRegex(ValueError, 'pos_encoding_size'):
        transformer_layers.TransformerBlocks.build_using_factories(
            self.hidden_size,
            intermediate_sizes=[],
            num_heads=2,
            dense_factory=self.dense_factory,
            use_additive_pos_encoding=True,
            rngs=nnx.Rngs(0),
        )

  def test_epd_transformer_with_additive_pos_encoding(self):
    """Tests EpdTransformer with positional encodings and pre-decode norm."""
    num_levels, pos_encoding_size = 16, 8
    input_size, output_size, latent_size = 6, 4, 12
    encode_factory = functools.partial(standard_layers.ConvLevel, kernel_size=1)
    decode_factory = functools.partial(standard_layers.ConvLevel, kernel_size=1)
    process_factory = functools.partial(
        transformer_layers.TransformerBlocks.build_using_factories,
        intermediate_sizes=(latent_size,),
        num_heads=2,
        dense_factory=self.dense_factory,
        relative_bias_factory=functools.partial(
            standard_layers.Mlp.uniform, hidden_size=4, hidden_layers=1
        ),
        pos_encoding_size=pos_encoding_size,
        pre_normalization_factory=nnx.LayerNorm,
        normalize_qk=True,
        gating=None,
    )
    epd = transformer_layers.EpdTransformer.build_using_factories(
        input_size=input_size,
        output_size=output_size,
        latent_size=latent_size,
        num_process_blocks=2,
        encode_factory=encode_factory,
        process_factory=process_factory,
        decode_factory=decode_factory,
        pos_encoding_size=pos_encoding_size,
        use_additive_pos_encoding=True,
        pre_decode_normalization_factory=nnx.LayerNorm,
        rngs=nnx.Rngs(0),
    )
    self.assertEmpty(nnx.find_duplicates(epd))
    self.assertIsInstance(epd.pos_encoding_proj, nnx.Linear)
    self.assertIsInstance(epd.pre_decode_norm, nnx.LayerNorm)
    x = jax.random.normal(jax.random.key(0), (input_size, num_levels))
    pe = jax.random.normal(jax.random.key(1), (pos_encoding_size, num_levels))
    out = epd(x, inputs_pos_encoding=pe)
    self.assertEqual(out.shape, (output_size, num_levels))
    with self.assertRaisesRegex(ValueError, 'inputs_pos_encoding is required'):
      epd(x)

    def loss_fn(model, inp):
      return jax.numpy.sum(model(inp, inputs_pos_encoding=pe) ** 2)

    grad_model, grad_x = nnx.grad(loss_fn, argnums=(0, 1))(epd, x)
    self.assertGreater(float(np.linalg.norm(grad_x)), 0.0)
    self.assertGreater(
        float(np.linalg.norm(grad_model.pos_encoding_proj.kernel.value)), 0.0
    )
    self.assertGreater(
        float(np.linalg.norm(grad_model.pre_decode_norm.scale.value)), 0.0
    )

  def test_epd_transformer_encodes_latents(self):
    """Tests EpdTransformer cross-attending to latents with their own size."""
    num_levels, num_latent_levels, pos_encoding_size = 16, 13, 8
    input_size, output_size, latent_size, latents_size = 6, 4, 12, 3
    embedding_factory = functools.partial(
        standard_layers.ConvLevel, kernel_size=1
    )
    process_factory = functools.partial(
        transformer_layers.TransformerBlocks.build_using_factories,
        intermediate_sizes=(latent_size,),
        num_heads=2,
        dense_factory=self.dense_factory,
        relative_bias_factory=functools.partial(
            standard_layers.Mlp.uniform, hidden_size=4, hidden_layers=1
        ),
        pos_encoding_size=pos_encoding_size,
        pre_normalization_factory=nnx.LayerNorm,
        gating=None,
    )
    epd = transformer_layers.EpdTransformer.build_using_factories(
        input_size=input_size,
        output_size=output_size,
        latent_size=latent_size,
        num_process_blocks=2,
        encode_factory=embedding_factory,
        process_factory=process_factory,
        decode_factory=embedding_factory,
        latents_size=latents_size,
        rngs=nnx.Rngs(0),
    )
    self.assertEmpty(nnx.find_duplicates(epd))
    self.assertIsInstance(epd.latents_encode, standard_layers.ConvLevel)
    x = jax.random.normal(jax.random.key(0), (input_size, num_levels))
    z = jax.random.normal(jax.random.key(1), (latents_size, num_latent_levels))
    x_pe = jax.random.normal(jax.random.key(2), (pos_encoding_size, num_levels))
    z_pe = jax.random.normal(
        jax.random.key(3), (pos_encoding_size, num_latent_levels)
    )
    out = epd(x, z, x_pe, z_pe)
    self.assertEqual(out.shape, (output_size, num_levels))
    with self.assertRaisesRegex(ValueError, 'latents are required'):
      epd(x, inputs_pos_encoding=x_pe)

    def loss_fn(model):
      return jax.numpy.sum(model(x, z, x_pe, z_pe) ** 2)

    grads = nnx.grad(loss_fn)(epd)
    self.assertGreater(
        float(np.linalg.norm(grads.latents_encode.kernel.value)), 0.0
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_shifts_even_window_shape_no_gating',
          input_size=6,
          output_size=3,
          n_layers=2,
          num_heads=2,
          inputs_window_shape=(2, 2),
          shift_windows=False,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='no_shifts_odd_window_shape_gating_where_possible',
          input_size=5,
          output_size=2,
          n_layers=1,
          inputs_window_shape=(3, 3),
          shift_windows=False,
          num_heads=3,
          gating=None,
      ),
      dict(
          testcase_name='shifts_even_window_shape_no_gating',
          input_size=6,
          output_size=3,
          n_layers=2,
          num_heads=2,
          inputs_window_shape=(2, 2),
          shift_windows=True,
          gating=lambda skip, x: x,
      ),
      dict(
          testcase_name='shifts_odd_window_shape_gating_where_possible',
          input_size=5,
          output_size=2,
          n_layers=1,
          inputs_window_shape=(3, 3),
          shift_windows=True,
          num_heads=3,
          gating=None,
      ),
  )
  def test_window_transformer_blocks_self_attention(
      self,
      input_size: int,
      output_size: int,
      n_layers: int,
      inputs_window_shape: tuple[int, int],
      shift_windows: bool,
      num_heads: int,
      gating: transformer_layers.Gating,
  ):
    """Tests output_shape of TransformerBlocks."""
    ylm_mapper = spherical_harmonics.YlmMapper()
    ylm_pe = transformer_layers.spherical_harmonic_lon_lat_encodings(
        ylm_mapper.ylm_map(self.grid), 4
    )
    rngs = nnx.Rngs(0)
    relative_bias_factory = functools.partial(
        standard_layers.Mlp, intermediate_sizes=()
    )
    net = transformer_layers.WindowTransformerBlocks.build_using_factories(
        input_size,
        output_size,
        intermediate_sizes=([self.hidden_size] * n_layers),
        num_heads=num_heads,
        relative_bias_factory=relative_bias_factory,
        pos_encoding_size=ylm_pe.shape[0],
        inputs_window_shape=inputs_window_shape,
        qkv_features=num_heads * self.per_head_kv_size,
        shift_windows=shift_windows,
        dense_factory=self.dense_factory,
        gating=gating,
        inputs_bc=boundaries.LonLatBoundary(),
        rngs=rngs,
    )
    self.assertEmpty(nnx.find_duplicates(net))
    rng = jax.random.key(0)
    inputs = jax.random.uniform(rng, (input_size,) + self.grid.shape)
    out = net(inputs, inputs_pos_encoding=ylm_pe)
    self.assertEqual(out.shape, (output_size,) + self.grid.shape)

  def tests_window_transformer_with_mask(self):
    """Tests output_shape of TransformerBlocks with mask."""
    ylm_mapper = spherical_harmonics.YlmMapper()
    ylm_pe = transformer_layers.spherical_harmonic_lon_lat_encodings(
        ylm_mapper.ylm_map(self.grid), 4
    )
    rngs = nnx.Rngs(0)
    num_heads = 2
    n_layers = 2
    inputs_window_shape = (2, 2)
    relative_bias_factory = functools.partial(
        standard_layers.Mlp, intermediate_sizes=()
    )
    input_size = 6
    output_size = 3
    net = transformer_layers.WindowTransformerBlocks.build_using_factories(
        input_size,
        output_size,
        intermediate_sizes=([self.hidden_size] * n_layers),
        num_heads=num_heads,
        relative_bias_factory=relative_bias_factory,
        pos_encoding_size=ylm_pe.shape[0],
        inputs_window_shape=inputs_window_shape,
        qkv_features=num_heads * self.per_head_kv_size,
        shift_windows=True,
        dense_factory=self.dense_factory,
        gating=lambda skip, x: x,
        inputs_bc=boundaries.LonLatBoundary(),
        rngs=rngs,
    )
    rng = jax.random.key(0)
    inputs = jax.random.uniform(rng, (input_size, *self.grid.shape))
    mask = jax.random.bernoulli(rng, 0.2, self.grid.shape)[np.newaxis, ...]
    out = net(inputs, inputs_pos_encoding=ylm_pe, mask=mask)
    self.assertEqual(out.shape, (output_size, *self.grid.shape))

  def test_spherical_harmonic_lon_lat_encodings_select_modes(self):
    """Tests that encodings are the harmonics with l_min <= l < l_max."""
    ylm_map = spherical_harmonics.YlmMapper().ylm_map(self.grid)
    l_min, l_max = 1, 4
    pe = transformer_layers.spherical_harmonic_lon_lat_encodings(
        ylm_map, l_max, l_min
    )
    self.assertEqual(pe.shape, (l_max**2 - l_min**2,) + self.grid.shape)
    # Each encoding is a single spherical harmonic with a selected wavenumber.
    modal = np.asarray(jax.vmap(ylm_map.to_modal_array)(pe))
    total_wavenumber = ylm_map.ylm_grid.fields['total_wavenumber'].data
    modes = set()
    for coeffs in modal:
      m_idx, l_idx = np.unravel_index(np.argmax(np.abs(coeffs)), coeffs.shape)
      self.assertBetween(total_wavenumber[l_idx], l_min, l_max - 1)
      modes.add((m_idx, l_idx))
    self.assertLen(modes, l_max**2 - l_min**2)
    with self.assertRaisesRegex(ValueError, 'does not resolve'):
      transformer_layers.spherical_harmonic_lon_lat_encodings(ylm_map, 10**3)

  def test_spherical_positional_encoder(self):
    """Tests output_shape of SphericalPositionalEncoder."""
    ylm_mapper = spherical_harmonics.YlmMapper()
    lmax = 21
    lmin = 0
    pe = transformer_layers.SphericalPositionalEncoder(ylm_mapper, lmax, lmin)
    grid = coordinates.LonLatGrid.T21()
    inputs = cx.field(np.ones(grid.shape), grid)
    pos_encoding = pe(inputs, ('longitude', 'latitude'))
    self.assertEqual(pos_encoding.shape, (lmax**2 - lmin**2,) + grid.shape)

  @parameterized.product(
      basis_type=['legendre', 'sinusoidal'],
      level_type=['sigma', 'hybrid', 'pressure', 'layer'],
  )
  def test_level_positional_encoder(self, basis_type: str, level_type: str):
    """Tests output shape and thickness features of LevelPositionalEncoder."""
    if level_type == 'sigma':
      levels = coordinates.SigmaLevels.equidistant(12)
    elif level_type == 'hybrid':
      levels = coordinates.HybridLevels.ecmwf137_interpolated(12)
    elif level_type == 'pressure':
      levels = coordinates.PressureLevels.with_13_era5_levels()
    else:
      levels = coordinates.LayerLevels(12)
    pe_size = 8
    pe = transformer_layers.LevelPositionalEncoder(
        encoding_size=pe_size,
        basis_type=basis_type,  # pyrefly: ignore[bad-argument-type]
    )
    inputs = cx.field(
        np.ones(levels.shape + self.grid.shape), levels, self.grid
    )
    pos_encoding = pe(inputs, (levels,), encoding_dim_tag='pe')
    self.assertEqual(pos_encoding.shape, (pe_size,) + levels.shape)
    self.assertEqual(pos_encoding.dims, ('pe', levels.dims[0]))
    self.assertTrue(np.all(np.isfinite(pos_encoding.data)))
    thickness = pos_encoding.data[:2]
    if level_type in ('sigma', 'layer'):  # equidistant layers.
      np.testing.assert_allclose(thickness, 0.0, atol=1e-5)
    else:  # stretched layers must be distinguishable through thickness.
      self.assertGreater(np.std(thickness[0]), 0.0)
      np.testing.assert_allclose(np.mean(thickness[1]), 0.0, atol=1e-5)
      np.testing.assert_allclose(np.std(thickness[1]), 1.0, atol=1e-5)
    if isinstance(levels, coordinates.HybridLevels):
      diffs = pos_encoding.data[:, :, None] - pos_encoding.data[:, None, :]
      channel_norms = np.linalg.norm(diffs, axis=(1, 2))
      self.assertTrue(np.all(channel_norms > 0.0))

  def test_level_fractions_and_thicknesses(self):
    """Tests column fractions for sigma, hybrid and pressure levels."""
    sigma = coordinates.SigmaLevels.equidistant(4)
    centers, thicknesses = transformer_layers.level_fractions_and_thicknesses(
        sigma
    )
    np.testing.assert_allclose(centers, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(thicknesses, 0.25)

    hybrid = coordinates.HybridLevels(
        a_boundaries=[0.0, 100.0, 0.0], b_boundaries=[0.0, 0.4, 1.0]
    )
    centers, thicknesses = transformer_layers.level_fractions_and_thicknesses(
        hybrid, reference_pressure=1000.0
    )
    np.testing.assert_allclose(centers, [0.25, 0.75])
    np.testing.assert_allclose(thicknesses, [0.5, 0.5])

    pressure = coordinates.PressureLevels([250.0, 500.0, 1000.0])
    centers, thicknesses = transformer_layers.level_fractions_and_thicknesses(
        pressure, reference_pressure=1000.0
    )
    np.testing.assert_allclose(centers, [0.25, 0.5, 1.0])
    # Interfaces at [0, 375, 750, 1250] hPa normalized by the bottom interface.
    np.testing.assert_allclose(thicknesses, [0.3, 0.3, 0.4])
    with self.assertRaisesRegex(ValueError, 'too small'):
      pe = transformer_layers.LevelPositionalEncoder(encoding_size=2)
      pe(cx.field(np.ones(sigma.shape), sigma), (sigma,), 'pe')

  def test_1d_encodings(self):
    """Tests Legendre and sinusoidal encodings against closed forms."""
    x = np.linspace(-1.0, 1.0, 7)
    legendre = transformer_layers.legendre_encodings(x, 3)
    np.testing.assert_allclose(legendre[0], x)
    np.testing.assert_allclose(legendre[1], (3 * x**2 - 1) / 2, atol=1e-6)
    np.testing.assert_allclose(legendre[2], (5 * x**3 - 3 * x) / 2, atol=1e-6)

    positions = np.array([0.0, 10.0, 100.0])
    sinusoidal = transformer_layers.sinusoidal_encodings(
        positions, 4, min_scale=1.0, max_scale=100.0
    )
    self.assertEqual(sinusoidal.shape, (4, 3))
    np.testing.assert_allclose(sinusoidal[0], np.sin(positions), atol=1e-6)
    np.testing.assert_allclose(
        sinusoidal[3], np.cos(positions / 100.0), atol=1e-6
    )
    with self.assertRaisesRegex(ValueError, 'even'):
      transformer_layers.sinusoidal_encodings(
          positions, 3, min_scale=1.0, max_scale=100.0
      )

  @parameterized.named_parameters(
      dict(testcase_name='soft_pooling', cross_attention=False),
      dict(testcase_name='perceiver_style', cross_attention=True),
  )
  def test_latent_pooling_transformer_over_spatial_grid(
      self, cross_attention: bool
  ):
    """Tests LatentPoolingTransformer on inputs with 2 spatial dimensions."""
    channels, num_latents, pos_encoding_size = 6, 5, 3
    spatial_shape = (4, 7)
    transformer_factory = functools.partial(
        transformer_layers.TransformerBlocks.build_using_factories,
        intermediate_sizes=(channels,),
        num_heads=2,
        dense_factory=self.dense_factory,
        gating=None,
    )
    cross_attention_factory = None
    if cross_attention:
      cross_attention_factory = functools.partial(
          transformer_layers.TransformerBlocks.build_using_factories,
          intermediate_sizes=(),
          num_heads=2,
          dense_factory=self.dense_factory,
          gating=None,
      )
    net = transformer_layers.LatentPoolingTransformer.build_using_factories(
        channels,
        channels,
        transformer_factory=transformer_factory,
        num_latents=num_latents,
        pos_encoding_size=pos_encoding_size,
        pool_transformer_factory=cross_attention_factory,
        unpool_transformer_factory=cross_attention_factory,
        rngs=nnx.Rngs(0),
    )
    self.assertEmpty(nnx.find_duplicates(net))
    self.assertEqual(net.num_latents, num_latents)
    self.assertEqual(net.pool_transformer is not None, cross_attention)
    self.assertEqual(net.unpool_transformer is not None, cross_attention)
    x = jax.random.normal(jax.random.key(0), (channels, *spatial_shape))
    pe = jax.random.normal(
        jax.random.key(1), (pos_encoding_size, *spatial_shape)
    )
    out = net(x, inputs_pos_encoding=pe)
    self.assertEqual(out.shape, (channels, *spatial_shape))
    with self.assertRaisesRegex(ValueError, 'does not support masks'):
      net(x, inputs_pos_encoding=pe, mask=np.ones((1, *spatial_shape)))
    with self.assertRaisesRegex(ValueError, 'inputs_pos_encoding is required'):
      net(x)
    with self.assertRaisesRegex(ValueError, 'do not match'):
      net(x, inputs_pos_encoding=pe[:, :2])

  def test_epd_transformer_with_latent_pooling(self):
    """Tests EpdTransformer with latent pooling blocks and relative bias."""
    num_levels, num_latents, pos_encoding_size = 16, 4, 8
    input_size, output_size, latent_size = 6, 4, 12
    transformer_factory = functools.partial(
        transformer_layers.TransformerBlocks.build_using_factories,
        intermediate_sizes=(latent_size,),
        num_heads=2,
        dense_factory=self.dense_factory,
        relative_bias_factory=functools.partial(
            standard_layers.Mlp.uniform, hidden_size=4, hidden_layers=1
        ),
        pos_encoding_size=pos_encoding_size,
        pre_normalization_factory=nnx.LayerNorm,
        normalize_qk=True,
        gating=None,
    )
    process_factory = functools.partial(
        transformer_layers.LatentPoolingTransformer.build_using_factories,
        transformer_factory=transformer_factory,
        num_latents=num_latents,
        pos_encoding_size=pos_encoding_size,
    )
    epd = transformer_layers.EpdTransformer.build_using_factories(
        input_size=input_size,
        output_size=output_size,
        latent_size=latent_size,
        num_process_blocks=2,
        encode_factory=functools.partial(
            standard_layers.ConvLevel, kernel_size=1
        ),
        process_factory=process_factory,
        decode_factory=functools.partial(
            standard_layers.ConvLevel, kernel_size=1
        ),
        pos_encoding_size=pos_encoding_size,
        use_additive_pos_encoding=True,
        pre_decode_normalization_factory=nnx.LayerNorm,
        rngs=nnx.Rngs(0),
    )
    self.assertEmpty(nnx.find_duplicates(epd))
    for block in epd.process_blocks:
      self.assertIsInstance(block, transformer_layers.LatentPoolingTransformer)
      self.assertEqual(block.num_latents, num_latents)
    x = jax.random.normal(jax.random.key(0), (input_size, num_levels))
    pe = jax.random.normal(jax.random.key(1), (pos_encoding_size, num_levels))
    out = epd(x, inputs_pos_encoding=pe)
    self.assertEqual(out.shape, (output_size, num_levels))

    def loss_fn(model, inp):
      return jax.numpy.sum(model(inp, inputs_pos_encoding=pe) ** 2)

    grad_model, grad_x = nnx.grad(loss_fn, argnums=(0, 1))(epd, x)
    self.assertGreater(float(np.linalg.norm(grad_x)), 0.0)
    grad_block = grad_model.process_blocks[0]
    self.assertGreater(
        float(np.linalg.norm(grad_block.pool_proj.kernel.value)), 0.0
    )
    self.assertGreater(
        float(np.linalg.norm(grad_block.unpool_proj.kernel.value)), 0.0
    )

  def test_pytree_no_duplicates(self):
    dummy_att = transformer_layers.MultiHeadAttention(
        num_heads=2,
        in_features=4,
        qkv_features=4,
        out_features=4,
        rngs=nnx.Rngs(0)
    )
    dummy_dense = standard_layers.Mlp.uniform(
        input_size=4,
        output_size=4,
        hidden_size=4,
        hidden_layers=1,
        rngs=nnx.Rngs(0)
    )
    block = transformer_layers.TransformerBlocks(
        attentions=[dummy_att],
        dense_layers=[dummy_dense],
        pre_norms=None,
        post_norms=None,
    )
    self.assertEmpty(nnx.find_duplicates(block))


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
