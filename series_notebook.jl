### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 33c00f16-714e-4c2e-8631-c5c0d2bb38ab
begin
	using Pkg
	#Pkg.add("ImageMagick")
	#Pkg.add("Images")
	#Pkg.add("Colors")
	Pkg.add("BenchmarkTools")
end

# ╔═╡ 9d9e5df3-1aa4-498b-9146-cc740d0b7fd7
begin
	using FFTW
	using ImageMagick
	using Images
	using Colors
	using PlutoUI
	using StaticArrays
	using Profile
	using BenchmarkTools
end

# ╔═╡ e6701b93-3804-455b-a7bf-9b581751431a
using Plots

# ╔═╡ bdebed31-102a-4031-998a-45d1372aafa7
md"""
maxiter
$(@bind maxiter Slider(1:100000))
"""

# ╔═╡ 0b514d03-264f-4d41-91b5-05f61fae5306
function moveSeries(coefficients, q)
	out::typeof(coefficients) = [0, 0, 0, 0]
	out[4] = coefficients[4]
	out[3] = coefficients[3] + 3 * out[4] * q
	out[2] = coefficients[2] + 2 * out[3] * q - 3 * out[4] * q^2
	out[1] = coefficients[1] +     out[2] * q -     out[3] * q^2 + out[4] * q^3
	return out
end

# ╔═╡ becc593b-7451-47bb-b8c2-f0f3d41053fb
begin
	function evaluatePolynomial(coefficients::SVector{4}, x)
		out::typeof(x) = 0
		for c = reverse(coefficients)
			out *= x
			out += c
		end
		return out
	end
	function evaluatePolynomial(coefficients, x)
		return evaluatePolynomial(SVector{4}(coefficients), x)
	end
end

# ╔═╡ 309d467c-d441-43b3-a55c-166596a9d777
(
	evaluatePolynomial([.1, 1, 4, 0.1], 2.),
	evaluatePolynomial(moveSeries([.1, 1, 4, 0.1], 2), -0.)
)

# ╔═╡ a500ce76-7a56-47a9-a755-9723362a4b22
begin
	function fourCorners(array)
		return (
			array[1:end ÷ 2, 1:end ÷ 2],
			array[1:end ÷ 2, end ÷ 2 + 1:end],
			array[end ÷ 2 + 1:end, 1:end ÷ 2],
			array[end ÷ 2 + 1:end, end ÷ 2 + 1:end],
			
		)
	end
	x = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
	size(fourCorners(CartesianIndices(x))[1]) == (2, 2)
end

# ╔═╡ f5756d28-ebc3-45bb-8202-0ca98c016578
begin
	function series_iterate(coefficients, center, maxdel, maxiters)
		A = coefficients[1]
		B = coefficients[2]
		C = coefficients[3]
		D = coefficients[4]
		imaxdel = im * maxdel
		inititers = 0
		while true
			abs(1000 * D * maxdel^3) <= abs(A + B * maxdel + C * maxdel^2) || break
			abs(1000 * D * imaxdel^3) <= abs(A + B * imaxdel + C * imaxdel^2) || break
			inititers < maxiters || break
			An = A^2 + center
			Bn = 2 * A * B + 1
			Cn = 2 * A * C + B^2
			Dn  = 2 * A * D + 2 * B * C
			A, B, C, D = An, Bn, Cn, Dn
			
			
			inititers += 1
		end
		return [A, B, C, D], inititers
	end
	function mandelbrot(center, radius, maxiters, res=1024)
		delta_arr = range(-radius, radius, length=res)
		delta_arr = delta_arr .+ im .* delta_arr' 
		
		coefficients::Array{Complex{Float64}, 1} = [0, 0, 0, 0]
		
		maxdel = maximum(abs.(delta_arr)) * 5
				
		coefficients, inititers = series_iterate(coefficients, center, maxdel, maxiters)
		A, B, C, D = coefficients
		
		z_init = A .+ B .* delta_arr .+ C .* delta_arr .^ 2 + D * delta_arr .^ 3
		
		out::Array{Int64, 2} = zeros(res, res)
		
		float_center = Complex{Float64}(center)
		for i = eachindex(out)
			c = float_center + delta_arr[i]
			count = inititers
			z = z_init[i]
			while count < maxiter && abs(z) < 10
				count = count + 1
				z *= z
				z += c
			end
			out[i] = count
		end
		return out, inititers
	end
	function outer_recursive_mandelbrot(center, radius, max_iters, res=1024)
		delta_arr = range(-radius, radius, length=res)
		delta_arr = delta_arr .+ im .* delta_arr' 
		c_arr = delta_arr .+ center
		coefficients::Array{Complex{Float64}, 1} = [0, 0, 0, 0]
		
		indices = CartesianIndices(c_arr)
		
		output_arr = zeros(Int64, size(c_arr))
		
		recursive_mandelbrot(
			indices, coefficients, center, c_arr, output_arr, 0, max_iters
		)
		return output_arr, 1
	end
	function recursive_mandelbrot(
			indices, coefficients, prev_center, c_arr, output_arr, init_iters, max_iters
		)
		
		center = (c_arr[indices[1]] + c_arr[indices[end]]) / 2
		coefficients = moveSeries(coefficients, center - prev_center)
		
		if size(indices) == (1, 1)
			count = init_iters
			z = coefficients[1]
			while count < max_iters && abs(z) < 10
				for dummy = 1:4
					count = count + 1
					z *= z
					z += center
				end
			end
			output_arr[indices[1]] = count
		else
			maxdel = abs(c_arr[indices[1]] - c_arr[indices[end]]) / 2
			coefficients, more_iters = series_iterate(coefficients, center, maxdel, max_iters - init_iters)
			init_iters += more_iters
			for sub_indices = fourCorners(indices)
				recursive_mandelbrot(sub_indices, coefficients, center, c_arr, output_arr, init_iters, max_iters)
			end
		end
	end	
end

# ╔═╡ da0b3816-e929-4454-8f1b-8d334d3ae93a
begin
	using JLD
	d = load("last_location.jld")
	out, inititers = outer_recursive_mandelbrot(
		Complex{Float64}(d["center"]), 
		d["radius"], 
		maxiter, 
		16 * 64
	)
	miniters = minimum(out)
	maxiters = maximum(out)
	out = out .- minimum(out)
	out = out ./ maximum(out)
	Gray.(out), inititers, miniters, maxiters
end

# ╔═╡ f1780d86-a0c4-4749-b067-793fd6574c56
begin
	Profile.clear()
	@Profile.profile outer_recursive_mandelbrot(
		Complex{Float64}(d["center"]), 
		d["radius"], 
		maxiter, 
		16 * 64
	)
end

# ╔═╡ 3b42fa11-5a61-4b89-811d-0e78b6a6ebfa
Profile.print()

# ╔═╡ 7bfbec42-e5cc-4537-8c74-01fbc16f5d0e
@btime outer_recursive_mandelbrot(
		Complex{Float64}(d["center"]), 
		d["radius"], 
		maxiter, 
		16 * 64
	)

# ╔═╡ Cell order:
# ╠═33c00f16-714e-4c2e-8631-c5c0d2bb38ab
# ╠═9d9e5df3-1aa4-498b-9146-cc740d0b7fd7
# ╠═f5756d28-ebc3-45bb-8202-0ca98c016578
# ╠═bdebed31-102a-4031-998a-45d1372aafa7
# ╠═da0b3816-e929-4454-8f1b-8d334d3ae93a
# ╠═e6701b93-3804-455b-a7bf-9b581751431a
# ╠═0b514d03-264f-4d41-91b5-05f61fae5306
# ╠═becc593b-7451-47bb-b8c2-f0f3d41053fb
# ╠═309d467c-d441-43b3-a55c-166596a9d777
# ╠═a500ce76-7a56-47a9-a755-9723362a4b22
# ╠═f1780d86-a0c4-4749-b067-793fd6574c56
# ╠═3b42fa11-5a61-4b89-811d-0e78b6a6ebfa
# ╠═7bfbec42-e5cc-4537-8c74-01fbc16f5d0e
