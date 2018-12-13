# telescope.jl

module telescope
using Plots
using Statistics, Distributions
using Images, ImageView, FileIO
using DelimitedFiles
using LsqFit
import Base.*
import Base.^

@. model(x,p) = p[1] * exp(-(x-p[2])^2/p[3]^2)
@. waist707(x,p) = p[1] * sqrt(1 + (((x-p[2])*707e-9)/(pi*p[1]^2))^2)
@. waist679(x,p) = p[1] * sqrt(1 + (((x-p[2])*679e-9)/(pi*p[1]^2))^2)


function readcsv(name)
	return convert(Array{Float64,2},readdlm(name,',')[:,1:end-1])
end

function gfit(name)
	dat = readcsv(name)
	Z = convert(Array{Float64,2},dat[:,1:end-1])
	Zx = vec(sum(Z,dims=1))
	Zy = vec(sum(Z,dims=2))
	mux = findmax(Zx)[2]
	muy = findmax(Zy)[2]
	fitx = curve_fit(model, 1.0:length(Zx), Zx, [500000.0,mux,200.0])
	fity = curve_fit(model, 1.0:length(Zy), Zy, [500000.0,muy,200.0])
	
	return fitx,fity
end

function diameter(name)
	fitx, fity = gfit(name)
	wx = fitx.param[3]
	wy = fity.param[3]
	return 2*wx, 2*wy
end

function showFit(name)
	dat = readcsv(name)
	Z = convert(Array{Float64,2},dat[:,1:end-1])
	Zx = vec(sum(Z,dims=1))
	Zy = vec(sum(Z,dims=2))
	mux = findmax(Zx)[2]*4.4e-6
	muy = findmax(Zy)[2]*4.4e-6
	fitx = curve_fit(model, 1.0:length(Zx), Zx, [500000.0,mux,200.0])
	fity = curve_fit(model, 1.0:length(Zy), Zy, [500000.0,muy,200.0])

	plot([Zx,model(1:length(Zx),fitx.param),Zy,model(1:length(Zy),fity.param)])
	println(fitx.param)
	println(fity.param)
end

function getProfiles(dr,name; plt=false)
	n = length(name)
	files = [i for i in readdir(dr) if (i[1:n]==name) & (i[end-3:end]==".csv")]
	dist = Array{Float64}(undef,0)
	wx = Array{Float64}(undef,0)
	wy = Array{Float64}(undef,0)
	converged = []
	for i in files
		println(i)
		push!(dist, parse(Float64, split(i[n+1:end],'_')[1]))
		fitx,fity = gfit(dr * i)
		push!(wx,fitx.param[3])
		push!(wy,fity.param[3])
		push!(converged,fitx.converged & fity.converged)
	end
	
	p = sortperm(dist)
	d = [i*.0254 for i in dist][p]
	WX = [i*4.4e-6 for i in wx][p]
	WY = [i*4.4e-6 for i in wy][p]
	return d,WX,WY,converged[p]
end

function findWaist(d::Array,w::Array,lambda)
	# Converts dist from inches to meters and wx from pixels to meters, and fits them to a Gaussian waist profile
	println(gEst(d,w))
	if lambda==679
		#fit = curve_fit(waist679,d,w,[min(w...) , d[findmin(w)[2]])
		fit = curve_fit(waist679,d,w,gEst(d,w))
	elseif lambda==707
		fit = curve_fit(waist707,d,w,gEst(d,w))
	end
	
	if fit.converged==false
		println("WARNING: Fit failed to converge.")
	end
	return fit.param
end

function gEst(d,w)
	# Estimates Gaussian waist parameters w, z0
	imax = findmax(w)[2]
	imin = findmin(w)[2]
	theta = 2*(w[imax]-w[imin])/(d[imax]-d[imin])
	return [abs((700e-9)/(pi*theta)), d[imin]-w[imin]/theta]
end

function findWaist(dr::String,name::String,lambda)
	dist,wx,wy,converged = getProfiles(dr,name)
	paramx = findWaist(dist,wx,lambda)
	paramy = findWaist(dist,wy,lambda)
	
	lambda==679 ? M = waist679 : M = waist707
	plot(dist,[wx,M(dist,paramx),wy,M(dist,paramy)])
	
	return paramx,paramy,dist,wx,wy,converged,M
end

#----------- Ray tracing ------------------------

mutable struct Lens
	f::Number
	z::Number
end

mutable struct Syst
	zRef::Number
	z0::Number
	w0::Number
	lambda::Number
	lenses::Array{Lens,1}
	qRef::Complex
	zR::Number
	function Syst(zRef::Number,z0::Number,w0::Number,lambda::Number,lenses::Array{Lens,1},qRef::Number,zR::Number)
		p = sortperm([l.z for l in lenses])
		return new(zRef,z0,w0,lambda,lenses[p],qRef,zR)
	end
end

Syst(zRef::Number,z0::Number,w0::Number,lambda::Number,lenses::Array{Lens,1}) = 
	Syst(zRef,z0,w0,lambda,lenses, (zRef-z0+im*pi*w0^2/lambda) ,pi*w0^2/lambda)
Syst(zRef::Number,z0::Number,w0::Number,lambda::Number) = 
	Syst(zRef,z0,w0,lambda,Array{Lens}(undef,0))

function sort!(s::Syst)
	p = sortperm([l.z for l in s.lenses])
	s.lenses = s.lenses[p]
end

function *(s::Syst,l::Lens)
	push!(s.lenses,l)
	sort!(s)
end

function (s::Syst)(z)
	L = [i for i in s.lenses if between(i.z,z,s.zRef)]
	if isempty(L)
		return s.qRef + z-s.zRef
	elseif z>s.zRef
		q = s.qRef
		x = s.zRef
		for i=1:length(L)
			q = (q+L[i].z-x)/(-q/L[i].f + 1 - (L[i].z-x)/L[i].f)
			x = L[i].z
		end
		return q + z-x
	else
		q = s.qRef
		x = s.zRef
		for i=length(L):-1:1
			q = (q+L[i].z-x)/(q/L[i].f + 1 + (L[i].z-x)/L[i].f)
			x = L[i].z
		end
		return q + z-x
	end
end

function between(a,b,c)
	return (a<b)&(c<a)|(a<c)&(b<a)
end

function ^(s::Syst,z::Number)
	q = s(z)
	x = real(q)
	zR = imag(q)
	return sqrt((1+(x/zR)^2)*zR*s.lambda/pi)
end

end