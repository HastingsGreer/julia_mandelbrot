using ImageMagick
using Images
using CUDAnative
using CUDAdrv
using CuArrays

using GtkReactive, Gtk.ShortNames
using Colors

function mandelbrot_seq(c, maxiter)
    res::Array{Complex{Float64}, 1} = []
    count = 0
    z::typeof(c) = 0.0
    push!(res, 0)
    while count < maxiter && abs(z) < 14000000
        count = count + 1
        z *= z
        z += c
        push!(res, z)
        
    end
    println("count", count)
    return res
end

function seq_search(c, radius, maxiter, seq_search_iters)
    
    x = mandelbrot_cu(c, radius * .66,
        maxiter, value(bitsig), 128, seq_search_iters)
    y = x .!= maximum(x)

    y = feature_transform(y)
    y = distance_transform(y)
    y = y ./ maximum(y)

    dc = range(-radius * .66, radius * .66, length=128)
    dc = dc .+ im .* dc'

    new_center = dc[argmax(y)]
    
    
    new_seq = mandelbrot_seq(new_center + c, maxiter)
       
    return new_seq, new_center
end

function mandelbrot_cu_kernel(delta_c_array, z_seq, out_array, maxiter)
    i = (blockIdx().x -1) * blockDim().x + threadIdx().x
    
    @inbounds delta_c = delta_c_array[i]
    
    count = 0
    
    l = length(z_seq)
    
    @inbounds delta_z = 0 * z_seq[1]
    @inbounds z_n = 0 * z_seq[1]
    while count < maxiter && count < l - 4 && abs(delta_z + z_seq[count + 1]) < 18
        count = count + 1
        @inbounds z_n = z_seq[count]
        delta_z = 2 * z_n * delta_z + delta_z^2 + delta_c
    end
    
    @inbounds out_array[i] = count
    
    return 
end

const ctx = CuContext(CuDevice(0))

const res = 512


function mandelbrot_cu(center, radius, maxiter, bit64, res=res, seq_search_iters=3)   
    
    c = range(-radius, radius, length=res)
    c = c .+ im .* c'
    
    if seq_search_iters != 0
        seq, new_center = seq_search(center, radius, maxiter, 
            seq_search_iters - 1)
    else
        seq = mandelbrot_seq(center, maxiter)
        new_center = 0
    end
    
    c = c .- new_center
    if bit64
        device_c = CuArray(Array{Complex{Float64}, 2}(c))
        device_seq = CuArray(Array{Complex{Float64}, 1}(seq))
    else
        device_c = CuArray(Array{Complex{Float32}, 2}(c))
        device_seq = CuArray(Array{Complex{Float32}, 1}(seq))
    end
    out::Array{Int64, 2} = zeros(res, res)
    
    device_out = CuArray(out)
    secs = CUDAdrv.@elapsed begin

        @cuda blocks=res threads=res mandelbrot_cu_kernel(device_c, device_seq, device_out, maxiter)
        synchronize(ctx)
        out = collect(device_out)
    end
    return out
end

centersig = Signal(Complex{Base.MPFR.BigFloat}(0. + 0.0im))
radiussig = Signal(2.)
itersig = Signal(10000)
bitsig = checkbox(;label="64 bit residuals")

bothsig = map(centersig, radiussig, itersig, bitsig) do center, radius, iter, bit64
    center, radius, iter, bit64
end

function normalize(v)
    return (v .- minimum(v)) ./ (maximum(v) - minimum(v) + .01)
end

imgsig = map(droprepeats(sampleon(every(0.02), bothsig))) do both
    center, radius, iter, bit64 = both
    out = mandelbrot_cu(center, radius, iter, bit64)
    
    out = out .% (iter - 1)
    
    #out = out .% 500
    #out = log.(out)
    
    #return normalize(out)
    
    return RGB.(normalize(out .% 6400), normalize(out .% 8000), normalize(out .% 10000))
    
    #color = applycolormap(mandelbrot_cu(center, radius, iter, bit64), cmap("L4"))
    #return RGB.(color[:,:,1], color[:,:,2], color[:, :, 3])
end

c = canvas(UserUnit, res, res)
win = Window(c)

auxwin = Window("Controls")
vbox = Box(:v)
push!(auxwin, vbox)
hbox = Box(:h)
l = Label("Log Iters")
push!(hbox, l)
n = slider(2:7)
adjiters = map(n) do val
    push!(itersig, 10^val)
end
push!(hbox, n)
push!(vbox, hbox)
zo = button("Zoom Out")
zou = map(zo) do val
    push!(radiussig, value(radiussig) * 6)
end
push!(vbox, zo)
push!(vbox, bitsig)
zr = button("Zoom Reset")
zru = map(zr) do val
    push!(centersig, 0)
    push!(radiussig, 2)
    push!(bitsig, false)
    push!(itersig, 10000)
end
push!(vbox, zr)

redraw = draw(c, imgsig) do cnvs, image
    copy!(cnvs, image)
end

hres = res / 2
clicksig = map(c.mouse.buttonpress) do btn
    offset = value(radiussig) * (
        ((btn.position.x - hres) / hres) * im + ((btn.position.y - hres) / hres)
    )
    push!(centersig, value(centersig) + offset)
    push!(radiussig, value(radiussig) / 6)
    println(value(radiussig))
end

push!(centersig, 0)
push!(radiussig, 2)

signal_connect(win, :destroy) do w
    destroy(auxwin)
end

Gtk.showall(win)
Gtk.showall(auxwin)

signal_connect(win, :destroy) do widget
    Gtk.gtk_quit()
end
Gtk.gtk_main()
destroy!(ctx)


