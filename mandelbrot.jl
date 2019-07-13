
using CUDAnative
using CUDAdrv
using CuArrays

using GtkReactive, Gtk.ShortNames
using PerceptualColourMaps, Colors

function mandelbrot_seq(c, maxiter)
    res::Array{Complex{Float64}, 1} = []
    count = 0
    z::typeof(c) = 0.0
    push!(res, 0)
    while count < maxiter && abs(z) < 14000000
        count = count + 1
        z = z^2 + c
        push!(res, z)
        
    end
    return res
end

function seq_search(c, radius, maxiter)
    seq = mandelbrot_seq(c, maxiter)
    new_center = 0
    for x in 1:30
        temp_center = radius * ((rand() - .5) * 2 + (rand() - .5) * 2im)
        temp_seq = mandelbrot_seq(temp_center + c, maxiter)
        if length(temp_seq) > length(seq)
            new_center = temp_center
            seq= temp_seq
        end
    end
    return seq, new_center
end

function mandelbrot_cu_kernel(delta_c_array, z_seq, out_array, maxiter)
    i = (blockIdx().x -1) * blockDim().x + threadIdx().x
    
    @inbounds delta_c = delta_c_array[i]
    
    count = 0
    
    l = length(z_seq)
    
    @inbounds delta_z = 0 * z_seq[1]
    @inbounds z_n = 0 * z_seq[1]
    while count < maxiter && count < l - 1 && abs(delta_z + z_seq[count + 1]) < 18
        count = count + 1
        @inbounds z_n = z_seq[count]
        delta_z = 2 * z_n * delta_z + delta_z^2 + delta_c
    end
    
    @inbounds out_array[i] = count
    
    return 
end

ctx = CuContext(CuDevice(0))

res = 1024

function mandelbrot_cu(center, radius, maxiter, bit64)   
    
    c = range(-radius, radius, length=res)
    c = c .+ im .* c'
    
    seq, new_center = seq_search(center, radius, maxiter)
    
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
bitsig = checkbox(;label="64 bit")

bothsig = map(centersig, radiussig, itersig, bitsig) do center, radius, iter, bit64
    center, radius, iter, bit64
end

function normalize(v)
    return (v .- minimum(v)) ./ (maximum(v) - minimum(v) + .01)
end

imgsig = map(droprepeats(sampleon(every(0.05), bothsig))) do both
    center, radius, iter, bit64 = both
    out = mandelbrot_cu(center, radius, iter, bit64)
    
    out = out .% maximum(out)
    
    out = out .% 500
    #out = log.(out)
    
    return normalize(out)
    
    #return RGB.(normalize(out .% 640), normalize(out .% 800), normalize(out .% 1000))
    
    color = applycolormap(mandelbrot_cu(center, radius, iter), cmap("L4"))
    return RGB.(color[:,:,1], color[:,:,2], color[:, :, 3])
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


