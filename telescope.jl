# telescope.jl
# Typical usage: 
	# You have a bunch of .csv files from a beam profiler in a directory called "C:\\example\\directory".
	# The csv files have names like "679data-2_0001.ascii.csv".
	# You make this call:
		# > Sx,Sy = makeSyst("C:\\example\\directory","679data-",679)
	# Then you make an array of lenses in the approximate position you want them:
		# > lenses = [Lens(.3,.5), Lens(.2,.7)]
	# Then you incorporate them into your optical system with appropriate modematching condition:
		# > zapLens!(Sx,zTarget,qTarget,lenses,lb,ub)

module telescope
using Plots
using Statistics, Distributions
using Images, ImageView, FileIO
using DelimitedFiles
using LsqFit
using Optim
import Base.*
import Base.^

plotly()
@. model(x,p) = p[1] * exp(-(x-p[2])^2/p[3]^2)
@. waist707(x,p) = p[1] * sqrt(1 + (((x-p[2])*707e-9)/(pi*p[1]^2))^2)
@. waist679(x,p) = p[1] * sqrt(1 + (((x-p[2])*679e-9)/(pi*p[1]^2))^2)


function readcsv(name)
	# This gets the csv's output by a Newport LBP2-VIS beam profiler and turns them into a Float64 array
		# readdlm is a function from DelimitedFiles that can parse a .csv
		# The last line in the beam profiler .csv's is an empty string, and gets thrown away
	return convert(Array{Float64,2},readdlm(name,',')[:,1:end-1])
end

function gfit(name,plt=false)
	# Fits a single image of a Gaussian beam to a Gaussian to extract the width
	# It actually fits the marginal sums to 1D Gaussians, so it cannot at the moment extract e.g. ellipticity
	# It includes heuristics for ensuring convergence
	dat = readcsv(name)
	Z = convert(Array{Float64,2},dat)
	Zx = vec(sum(Z,dims=1))			# Marginal x sum
	Zy = vec(sum(Z,dims=2))
	Mx = findmax(Zx)					# [maximum value, maximum location] for Zx
	My = findmax(Zy)
	sigx = sum(Zx)/(sqrt(pi)*Mx[1])		# Estimate for width
	sigy = sum(Zy)/(sqrt(pi)*My[1])
	fitx = curve_fit(model, 1.0:length(Zx), Zx, [Mx[1],Mx[2],sigx])
	fity = curve_fit(model, 1.0:length(Zy), Zy, [My[1],My[2],sigy])
	
	if plt
		plot([Zx,model(1:length(Zx),fitx.param),Zy,model(1:length(Zy),fity.param)])
		return [Zx,model(1:length(Zx),fitx.param),Zy,model(1:length(Zy),fity.param)]
	end
	return fitx,fity
end

function diameter(name)
	# Finds the beam diameter given an image
	fitx, fity = gfit(name)
	wx = fitx.param[3]
	wy = fity.param[3]
	return 2*wx, 2*wy
end

function showFit(name)
	# Plots the image marginals along with fitted function for visual comparison
	dat = readcsv(name)
	Z = convert(Array{Float64,2},dat)
	Zx = vec(sum(Z,dims=1))			# Marginal x sum
	Zy = vec(sum(Z,dims=2))
	Mx = findmax(Zx)					# [maximum value, maximum location] for Zx
	My = findmax(Zy)
	sigx = sum(Zx)/(sqrt(pi)*Mx[1])		# Estimate for width
	sigy = sum(Zy)/(sqrt(pi)*My[1])
	fitx = curve_fit(model, 1.0:length(Zx), Zx, [Mx[1],Mx[2],sigx])
	fity = curve_fit(model, 1.0:length(Zy), Zy, [My[1],My[2],sigy])
	
	return [Zx,model(1:length(Zx),fitx.param),Zy,model(1:length(Zy),fity.param)]
	plot([Zx,model(1:length(Zx),fitx.param),Zy,model(1:length(Zy),fity.param)])
	println(fitx.param)
	println(fity.param)
end

function getProfiles(dr,name; plt=false, lunit=.0254, wunit=4.4e-6)
	# Gets fit data (beam diameters) for all files in a directory.
	# Assumes that file names have the format dr*name*"distance"*"cruff"*".csv"
		# Where all are strings, * is concatenation, and "distance" is an actual distance of the image
		# from some reference point.  "distance" is extracted and returned as relevant data.
	# lunit and wunit are unit conversion factors for the distances (lenghts) and widths, respectively.
		# The default values convert inch distances and pixel widths into meters.
	n = length(name)
	files = [i for i in readdir(dr) if (i[1:n]==name) & (i[end-3:end]==".csv")]	# Get all csv's in directory
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
	d = [i*lunit for i in dist][p]
	WX = [i*wunit for i in wx][p]
	WY = [i*wunit for i in wy][p]
	return d,WX,WY,converged[p]
end

function findWaist(d::Array,w::Array,lambda)
	# Converts dist from inches to meters and wx from pixels to meters, and fits them to a Gaussian waist profile.
	println("Estimated waist parameters: $(gEst(d,w))")
	
	@. waist(x,p) = p[1] * sqrt(1 + (((x-p[2])*lambda)/(pi*p[1]^2))^2)
	fit = curve_fit(waist,d,w,gEst(d,w))
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
	# Gets beam data from file, then uses it to find a waist
	dist,wx,wy,converged = getProfiles(dr,name)
	paramx = findWaist(dist,wx,lambda)
	paramy = findWaist(dist,wy,lambda)
	return paramx,paramy,dist,wx,wy,converged,M
end

function makeSyst(dr::String,name::String,lambda)
	dat = findWaist(dr::String,name::String,lambda)
	wx = dat[1][1]
	zx = dat[1][2]
	wy = dat[2][1]
	zy = dat[2][2]
	
	return Syst(zx,zx,wx,lambda), Syst(zy,zy,wy,lambda)
end

#----------- Ray tracing ------------------------

# The structure of this ray tracing business is organized by lenses and systems.  
# A system is a collection of lenses and a single beam

mutable struct Lens
	f::Number
	z::Number
end

mutable struct Syst
	zRef::Number	# Location from which to start propagating the beam
	z0::Number	# Beam waist location wrt to zRef in absence of any lenses
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

function clear!(s::Syst)
	s.lenses = Array{Lens}(undef,0)
end

function (s::Syst)(z::Number)
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

function (s::Syst)(z::Array)
	return [s(i) for i in z]
end

function ^(s::Syst,z::Number)
	# Computes the beam waist at any location z.
	q = s(z)
	x = real(q)
	zR = imag(q)
	return sqrt((1+(x/zR)^2)*zR*s.lambda/pi)
end

function ^(s::Syst,z::Array)
	return [s^i for i in z]
end

function overlap(q1,q2)
	# Computes the overlap integral for two beams with q parameters q1 and q2
	zR1 = imag(q1)
	zR2 = imag(q2)
	return 2*sqrt(zR1*zR2)/abs(q1-conj(q2))
end

function zapLens!(s::Syst,zTarget,qTarget,lenses::Array{Lens,1},lb,ub)
	# Optimizes the z locations of the lenses in "lenses" and adds them to System s.
	# The optimization criterion is mode matching to qTarget at location zTarget
	N = length(lenses)
	for l in lenses
		s*l
	end
	initial = [l.z for l in lenses]
	function objective!(x)
		if length(x)!=N
			error("Input must have length $N.  It has length $(length(x)) instead.")
		end
		for i in 1:N
			lenses[i].z = x[i]
		end
		sort!(s)
		return -overlap(s(zTarget),qTarget)
	end
	inner_optimizer = GradientDescent()
	results = optimize(objective!,lb,ub,initial, Fminbox(inner_optimizer))
	#results = optimize(objective!,initial)
	for i in 1:N
		lenses[i].z = results.minimizer[i]
	end
	sort!(s)
	return results
end

end