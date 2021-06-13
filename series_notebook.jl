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

# ╔═╡ 9d9e5df3-1aa4-498b-9146-cc740d0b7fd7
begin
	using FFTW
	using ImageMagick
	using Images
	using Colors
	using PlutoUI
end

# ╔═╡ 33c00f16-714e-4c2e-8631-c5c0d2bb38ab


# ╔═╡ bdebed31-102a-4031-998a-45d1372aafa7
md"""
maxiter
$(@bind maxiter Slider(1:100000))
"""

# ╔═╡ f5756d28-ebc3-45bb-8202-0ca98c016578
begin
	function mandelbrot(center, radius, maxiters, res=1024)
		delta_arr = range(-radius, radius, length=res)
		delta_arr = delta_arr .+ im .* delta_arr' 
		
		A = 0
		B = 0
		C = 0
		D = 0
		
		maxdel = maximum(abs.(delta_arr))
		imaxdel = im * maxdel
		inititers = 0
		while true
			An = A^2 + center
			Bn = 2 * A * B + 1
			Cn = 2 * A * C + B^2
			Dn  = 2 * A * D + 2 * B * C
			A, B, C, D = An, Bn, Cn, Dn
			
			abs(100 * D * maxdel^3) < abs(A + B * maxdel + C * maxdel^2) || break
			abs(100 * D * imaxdel^3) < abs(A + B * imaxdel + C * imaxdel^2) || break
			inititers += 1
		end
		
		print(D)
		A, B, C, D = Complex{Float64}.((A, B, C, D))
		
		z_init = A .+ B .* delta_arr .+ C .* delta_arr .^ 2 + D * delta_arr .^ 3
		
		out::Array{Int64, 2} = zeros(res, res)
		
		float_center = Complex{Float64}(center)
		Threads.@threads for i = eachindex(out)
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
	function mandelbrot2(center, radius, maxiters, res=1024)
		c_arr = range(-radius, radius, length=res)
		c_arr = c_arr .+ im .* c_arr' .+ center
		out::Array{Int64, 2} = zeros(res, res)
		for i = eachindex(out)
			c = c_arr[i]
			count = 0
			z::typeof(c) = 0.0
			while count < maxiter && abs(z) < 10
				count = count + 1
				z *= z
				z += c
			end
			out[i] = count
		end
		return out
	end
end

# ╔═╡ da0b3816-e929-4454-8f1b-8d334d3ae93a
begin
	using JLD
	d = load("last_location.jld")
	out, inititers = mandelbrot(
		Complex{Float64}(d["center"]), 
		d["radius"], 
		maxiter, 
		364
	)
	miniters = minimum(out)
	maxiters = maximum(out)
	out = out .- minimum(out)
	out = out ./ maximum(out)
	Gray.(out), inititers, miniters, maxiters
end

# ╔═╡ e6701b93-3804-455b-a7bf-9b581751431a
md"""
Initialization Iterations
$(@bind inititer Slider(0:100000))
"""

# ╔═╡ 0b514d03-264f-4d41-91b5-05f61fae5306
typeof(Complex{Float64}(d["center"]))

# ╔═╡ Cell order:
# ╠═33c00f16-714e-4c2e-8631-c5c0d2bb38ab
# ╠═9d9e5df3-1aa4-498b-9146-cc740d0b7fd7
# ╠═f5756d28-ebc3-45bb-8202-0ca98c016578
# ╠═da0b3816-e929-4454-8f1b-8d334d3ae93a
# ╠═bdebed31-102a-4031-998a-45d1372aafa7
# ╠═e6701b93-3804-455b-a7bf-9b581751431a
# ╠═0b514d03-264f-4d41-91b5-05f61fae5306
